mod image_processor;

use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
use std::error::Error;
use std::{fs, io};
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    println!("***************************************************************************************");
    println!("[이미지 배경 제거 프로그램] ");
    println!("VERSION : 1.0.0 ");
    println!("input 디렉토리에 있는 .jpg .png 이미지 파일들의 배경 제거하여 output 디렉토리에 저장합니다.");
    println!("* 인물 이미지만 잘 작동합니다");
    println!("***************************************************************************************\n");

    let input_dir = Path::new("input");
    let output_dir = Path::new("output");

   // 디렉토리 확인 및 생성
    if !input_dir.exists() {
      println!("디렉토리 '{}'가 존재하지 않습니다. 'input' 디렉토리를 생성하세요.", input_dir.display());
      return Ok(());
    }
   
    if !output_dir.exists() {
      fs::create_dir(output_dir)?;
    }
    // input 디렉토리에 파일이 있는지 확인
    let files: Vec<_> = fs::read_dir(input_dir)?.filter_map(|entry| entry.ok()).collect();

    if files.is_empty() {
        println!("'input' 디렉토리에 파일이 없습니다. 이미지를 추가한 후 다시 실행하세요.");
        return Ok(());
    }
    println!("input 디렉토리에 총 {}개의 이미지가 발견되었습니다.", files.len());
    println!("작업을 수행하시려면 엔터 키를 눌러주세요...");

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("입력을 읽는 데 실패했습니다.");


    // ONNX Runtime 환경 생성
    let environment = Environment::builder()
        .with_name("remove_background") // 환경 이름 설정
        .with_log_level(LoggingLevel::Warning) // 로그 레벨 설정
        .build()?;

    // 세션 생성
    let mut session: onnxruntime::session::Session<'_> = environment
        .new_session_builder()? // GPU 가속은 일단 사용안함
        .with_optimization_level(GraphOptimizationLevel::Basic)? // 최적화 레벨 : 기본
        .with_model_from_file("modnet.onnx")
        .expect("** 모델 로딩에 실패하였습니다. modnet.onnx 파일이 .exe 파일 경로에 존재하는지 확인해주세요**"); 
        
    println!("모델이 성공적으로 로드되었습니다!\n");
    
    println!("************* 이미지 배경제거 작업을 시작합니다 *************");
     // 디렉토리 순회하며 배경제거 수행
    process_directory(input_dir, output_dir, &mut session)?;

    println!("모든 처리가 완료되었습니다. 'output' 디렉토리를 확인하세요.");
    println!("프로그램을 종료하시려면 엔터 키를 누르세요...");

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("입력을 읽는 데 실패했습니다.");
    println!("엔터 키가 입력되었습니다. 프로그램을 종료합니다.");
    Ok(())
}

fn process_directory(
    input_dir: &Path,
    output_dir: &Path,
    session: &mut onnxruntime::session::Session<'_>,
) -> std::io::Result<()> {
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {

            // 파일 확장자 확인 (이미지 파일만 처리)
            if let Some(ext) = path.extension() {
                if ext == "jpg" || ext == "png" {
                    let output_file = output_dir.join(path.file_name().unwrap());
                    match process_image(&path, &output_file, session) {
                        Ok(_) => println!("처리 완료: {:?} -> {:?}\n", path.display(), output_file),
                        Err(e) => println!("처리 중 오류 발생: {:?}, {}", path.display(), e),
                    }
                }
            }
        }
    }
    Ok(())
}

fn process_image(
    input_path: &Path,
    output_path: &Path,
    session: &mut onnxruntime::session::Session<'_>,
) -> Result<(), Box<dyn Error>> {

    let input_tensor = image_processor::preprocess_image(input_path.to_str().unwrap())?;
    // 이미지 크기 가져오기
    let width = input_tensor.shape()[3] as usize;
    let height = input_tensor.shape()[2] as usize;
    println!("입력 이미지: {:?} ({}x{})", input_path.display(), width, height);

    // ONNX 추론 수행
    let outputs: Vec<OrtOwnedTensor<f32, _>> = session.run(vec![input_tensor.into()])?;
    let mask_tensor = outputs[0].view().into_shape((1, 1, height, width))?;

    // 배경제거 후 결과 저장
    let maked_image = image_processor::apply_mask(input_path.to_str().unwrap(), mask_tensor.to_owned())?;
    maked_image.save(output_path)?;
    Ok(())
}
