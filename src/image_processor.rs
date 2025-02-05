use std::error::Error;
use onnxruntime::ndarray::{self, Array4, ArrayView2};
use image::{open, ImageBuffer, Rgba, RgbaImage};
use std::path::Path;
use onnxruntime::tensor::OrtOwnedTensor;
use ndarray::{Array2, Axis};
use fast_image_resize::{CpuExtensions, PixelType, Resizer};
use fast_image_resize::images::Image;
use rayon::prelude::*;

const REF_SIZE: u32 = 512; 

pub fn process_image(
    input_path: &Path,
    output_path: &Path,
    session: &mut onnxruntime::session::Session<'_>,
) -> Result<(), Box<dyn Error>> {

    let img :ImageBuffer<image::Rgba<u8>, Vec<u8>> = open(input_path)?.into_rgba8(); 
    let (width, height) = img.dimensions();

    println!("> 입력 이미지 : {:?} ({}x{})", input_path.display(), width, height);

    let input_tensor = preprocess_image(&img, width, height)?;
    // let input_tensor = image_processor::preprocess_image(input_path.to_str().unwrap())?;

    // 이미지 크기 가져오기
    let re_width = input_tensor.shape()[3] as usize;
    let re_height = input_tensor.shape()[2] as usize;

    // ONNX 추론 수행
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![input_tensor.into()])?;
    let mask_tensor = outputs[0].view().into_shape((1, 1, re_height, re_width))?;

    
    let resized_mask = if re_width == width as usize && re_height == height as usize {
        // 크기가 동일하면 기존 마스크 텐서를 그대로 사용
        mask_tensor.to_owned()
    } else {
        // 크기가 다르면 리사이즈 수행
        resize_mask(mask_tensor.index_axis(Axis(0), 0)
                                .index_axis(Axis(0), 0), 
                    width, height)
    };
    
    
    // 배경제거 후 결과 저장
    let maked_image = apply_mask(input_path.to_str().unwrap(), resized_mask)?;
  
    // 파일 확장자 확인 후 JPEG의 경우 RGB 변환
    let mut output_path = output_path.to_path_buf(); // PathBuf로 변환
    output_path.set_extension("png"); // 확장자를 항상 "png"로 설정
    
    maked_image.save(output_path)?; // 항상 PNG로 저장
    
    Ok(())
}


/**
 * 빠른 리사이즈 함수 (SIMD 최적화)
 * 
 * 개발 디버깅시는 안빠른데, 릴리즈 빌드하면 빠름
 * OpenCV 리사이즈 보다 빠른지는 비교실험 필요
 * 
 * SIMD 최적화를 하려면 U8x4 연산을 해야함
 * 
 */
pub fn fast_resize(img: &ImageBuffer<Rgba<u8>, Vec<u8>>, new_width: u32, new_height: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();
    let raw_data = img.as_raw().to_vec(); // 메모리 복사 최소화

    if width == new_width && height == new_height {
        return img.clone(); // 크기가 같으면 원본 이미지 그대로 반환
    }

    // `fast_image_resize`용 이미지 데이터 변환 (RGBA: U8x4)
    let src_image = Image::from_vec_u8(width, height, raw_data, PixelType::U8x4)
                                            .unwrap();

    let mut dst_image = Image::new(new_width, new_height, PixelType::U8x4);
    let mut resizer = Resizer::new();

    let detected_extensions = CpuExtensions::default(); // CPU가 지원하는 SIMD 자동 감지
    unsafe { resizer.set_cpu_extensions(detected_extensions) };
    
    let _ = resizer.resize(&src_image, &mut dst_image, None);

    let raw_pixels = dst_image.buffer().to_vec();
    ImageBuffer::from_raw(new_width, new_height, raw_pixels)
                    .expect("Failed to create ImageBuffer")
}


/**
 * 이미지 전처리 함수 (RGB → BCHW 변환)
 
 - ModNet 최적화 : 평균 (0.5, 0.5, 0.5), 표준편차 (0.5, 0.5, 0.5) 정규화

 */
fn preprocess_image(
    img: &ImageBuffer<Rgba<u8>, Vec<u8>>, 
    width: u32, 
    height: u32
) -> Result<Array4<f32>, Box<dyn Error>> {

    let (re_width, re_height) = resize_dimensions(width, height, REF_SIZE);

    // `fast_resize()` 중복 방지: 크기가 같으면 원본 이미지 그대로 사용
    let resized_img = fast_resize(img, re_width, re_height);

    let buffer = resized_img.as_raw();
    let mut input_data = vec![0.0; (re_width * re_height * 3) as usize];
    // RGB 데이터를 (C, H, W) 형태로 변환
    for (i, pixel) in resized_img.pixels().enumerate() {
        // 인덱스 오버플로우 방지
        if i >= buffer.len() {
            break;
        }
        input_data[i] = (pixel[0] as f32 / 255.0 - 0.5) / 0.5; // R
        input_data[i + (re_width * re_height) as usize] = (pixel[1] as f32 / 255.0 - 0.5) / 0.5; // G
        input_data[i + 2 * (re_width * re_height) as usize] = (pixel[2] as f32 / 255.0 - 0.5) / 0.5; // B
    }
    // ONNX 모델 입력 차원 (Batch, Channels, Height, Width) 형태로 변환
    let input_tensor = Array4::from_shape_vec((1, 3, re_height as usize, re_width as usize), input_data)?;
    Ok(input_tensor)
}


/**
 * 입력 차원 조정 함수
 
 - MODNet 모델은 32의 배수 크기만 입력 가능하므로 입력 이미지 크기를 32의 배수로 조정해야 함
  - 가로/세로 중 큰 쪽을 ref_size로 맞추고, 비율을 유지한 상태에서 32의 배수로 조정.

 */
fn resize_dimensions(width: u32, height: u32, ref_size: u32) -> (u32, u32) {
    let max_size = width.max(height); // ✅ 큰 값 선택

    let (mut new_width, mut new_height) = if max_size > ref_size {
        if width >= height {
            (ref_size, (height as f32 * (ref_size as f32 / width as f32)) as u32)
        } else {
            ((width as f32 * (ref_size as f32 / height as f32)) as u32, ref_size)
        }
    } else {
        (width, height) // 크기가 ref_size보다 작으면 원본 유지
    };

    // 32의 배수로 조정
    new_width = (new_width / 32) * 32;
    new_height = (new_height / 32) * 32;

    (new_width, new_height)
}


/// 마스크 리사이즈 후 `Array4<f32>` 반환
pub fn resize_mask(mask_tensor: ArrayView2<f32>, orig_width: u32, orig_height: u32) -> Array4<f32> {
    let (mask_width, mask_height) = (mask_tensor.shape()[1] as u32, mask_tensor.shape()[0] as u32);

    // `fast_image_resize`용 `GrayImage` 변환
    let raw_data: Vec<u8> = mask_tensor.iter()
        .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
        .collect();

    let src_image = Image::from_vec_u8(mask_width, mask_height, raw_data, PixelType::U8)
        .expect("Failed to create source image");

    let mut dst_image = Image::new(orig_width, orig_height, PixelType::U8);
    let mut resizer = Resizer::new();

    let detected_extensions = CpuExtensions::default(); // CPU에서 지원하는 SIMD 자동 감지
    unsafe { resizer.set_cpu_extensions(detected_extensions) };
    
    resizer.resize(&src_image, &mut dst_image, None)
        .expect("Resizing failed");

    // `fast_image_resize` 결과를 `ndarray::Array2<f32>`로 변환
    let resized_mask = Array2::<f32>::from_shape_vec(
        (orig_height as usize, orig_width as usize),
        dst_image.buffer().iter().map(|&v| v as f32 / 255.0).collect()
    ).expect("Failed to convert resized image to ndarray");

    // (1, 1, height, width) 형태로 변환
    resized_mask.insert_axis(Axis(0)).insert_axis(Axis(0))
}


/**
 * 마스크 적용하여 이미지 배경제거
 */
pub fn apply_mask(image_path: &str, mask: Array4<f32>) -> Result<RgbaImage, Box<dyn Error>> {
    let img = open(image_path)?.into_rgba8();
    let (width, height) = img.dimensions();
    let mask_data = mask.index_axis(ndarray::Axis(0), 0).to_owned();

    let mut output_img = RgbaImage::new(width, height);
    let pixels = output_img.as_mut();

    // 픽셀 단위 병렬 처리
    pixels.par_chunks_exact_mut(4).enumerate().for_each(|(i, chunk)| {
        let x = (i % width as usize) as u32;
        let y = (i / width as usize) as u32;
        let pixel = img.get_pixel(x, y);
        let alpha = (mask_data[[0, y as usize, x as usize]] * 255.0) as u8;

        chunk[0] = pixel[0]; // R
        chunk[1] = pixel[1]; // G
        chunk[2] = pixel[2]; // B
        chunk[3] = alpha;    // Alpha (마스크 적용)
    });

    Ok(output_img)
}


