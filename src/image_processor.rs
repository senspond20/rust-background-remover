use std::error::Error;
use onnxruntime::ndarray::{self, Array4};
use image::{open, RgbaImage};

/**
 * 이미지 전처리 함수 (RGB → BCHW 변환)
 */
pub fn preprocess_image(image_path: &str) -> Result<Array4<f32>, Box<dyn Error>> {
    let img = open(image_path)?.into_rgb8(); // RGBA → RGB 변환
    let (width, height) = img.dimensions();
    let mut input_data = vec![0.0; (width * height * 3) as usize];

    // RGB 데이터를 (C, H, W) 형태로 변환
    for (i, pixel) in img.pixels().enumerate() {
        input_data[i] = pixel[0] as f32 / 255.0; // R
        input_data[i + (width * height) as usize] = pixel[1] as f32 / 255.0; // G
        input_data[i + 2 * (width * height) as usize] = pixel[2] as f32 / 255.0; // B
    }

    // ONNX 모델 입력 차원 (Batch, Channels, Height, Width) 형태로 변환
    let input_tensor = Array4::from_shape_vec((1, 3, height as usize, width as usize), input_data)?;
    Ok(input_tensor)
}

/**
 * 마스크 적용하여 이미지 배경제거
 */
pub fn apply_mask(image_path: &str, mask: Array4<f32>) -> Result<RgbaImage, Box<dyn Error>> {
    let img = open(image_path)?.into_rgba8();
    let (width, height) = img.dimensions();
    let mask_data = mask.index_axis(ndarray::Axis(0), 0).to_owned();
    
    let mut output_img = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            let alpha = (mask_data[[0, y as usize, x as usize]] * 255.0) as u8;
            output_img.put_pixel(x, y, image::Rgba([pixel[0], pixel[1], pixel[2], alpha]));
        }
    }
    Ok(output_img)
}