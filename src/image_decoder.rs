use std::fs::File;
use std::io::Read;
use image::DynamicImage;
use webp::Decoder;

/**
 * WebP 이미지를 DynamicImage로 변환
 */
pub fn convert_webp(input_path: &str) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    // WebP 파일 읽기
    let mut webp_data = Vec::new();
    let mut input_file = File::open(input_path)?;
    input_file.read_to_end(&mut webp_data)?;

    // WebP 디코딩
    let decoder = Decoder::new(&webp_data);
    let webp_image = match decoder.decode() {
        Some(img) => img,
        None => return Err("Failed to decode WebP image".into()),
    };

    // WebP 이미지를 DynamicImage로 변환
    let webp_image = webp_image.to_image();
    let dynamic_image = DynamicImage::ImageRgba8(webp_image.into());
    Ok(dynamic_image)
}


// /**
//  * AVIF 이미지를 DynamicImage로 변환
//  */
// pub fn convert_avif(input_path: &str, output_path: &str) -> Result<DynamicImage, Box<dyn std::error::Error>> {
//     // AVIF 파일 읽기
//     let mut avif_data = Vec::new();
//     let mut file = File::open(input_path)?;
//     file.read_to_end(&mut avif_data)?;

//     // AVIF 디코딩
//     let avif_image = Decoder::new(&avif_data)?.decode()?;
//     let dynamic_image = DynamicImage::ImageRgba8(avif_image);
//     Ok(dynamic_image)
// }
