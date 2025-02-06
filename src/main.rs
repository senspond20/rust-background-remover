mod image_processor;
mod image_decoder;
use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
use std::error::Error;
use std::path::Path;
use std::{fs, io};
use std::time::Instant;
use num_cpus;

const SUPPORTED_EXTS: [&str; 4] = ["jpg", "jpeg", "png",  "webp"];
// "avif" Cmake 빌드 필요

fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "*".repeat(90));
    println!("[이미지 배경 제거 프로그램]");
    println!("- 제작자 : RGB코딩");
    println!("- Version : 1.0.2\n");
    println!("[프로그램 소개]");
    println!("input 디렉토리에 있는 이미지 파일들의 배경 제거하여 output 디렉토리에 저장합니다.");
    println!("* 지원 파일 포맷 : {}", SUPPORTED_EXTS.join(", "));
    println!("* 인물 이미지만 잘 작동합니다\n");
    println!("* 공유는 자유지만, 원작자 표기는 부탁드립니다 ");
    println!("👉 https://www.youtube.com/@rgbitcode");
    println!("{}\n", "*".repeat(90));

    let input_dir = Path::new("input");
    let temp_dir = Path::new("temp");
    let output_dir = Path::new("output");

   // 디렉토리 확인 및 생성
    if !input_dir.exists() {
      println!("디렉토리 '{}'가 존재하지 않습니다. 'input' 디렉토리를 생성하세요.", input_dir.display());
      return Ok(());
    }
   
    if !output_dir.exists() {
      fs::create_dir(output_dir)?;
    }
    if !temp_dir.exists() {
      fs::create_dir(temp_dir)?;
    }
    // input 디렉토리에 파일이 있는지 확인
    let files: Vec<_> = fs::read_dir(input_dir)?.filter_map(|entry| entry.ok()).collect();

    if files.is_empty() {
        println!("'input' 디렉토리에 파일이 없습니다. 이미지를 추가한 후 다시 실행하세요.");
        return Ok(());
    }
    let supported_files: Vec<_> = files.iter()
    .filter(|entry| {
        if let Some(ext) = entry.path().extension() {
            SUPPORTED_EXTS.contains(&ext.to_str().unwrap_or("").to_lowercase().as_str())
        } else {
            false
        }
    })
    .collect();

    println!(
        "input 디렉토리에 총 {}개의 파일이 발견되었으며, 그 중 {}개가 지원되는 파일입니다.",
        files.len(),
        supported_files.len()
    );
    println!("작업을 수행하시려면 엔터 키를 눌러주세요...");

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("입력을 읽는 데 실패했습니다.");

    let logical_cpus = num_cpus::get();
    let n_threads : i16 = if logical_cpus > 4 { // CPU 논리적 코어 4개 이상이면 
        (logical_cpus as f64 * 0.75).round() as i16 // 논리적 코어의 75%
    } else {
        logical_cpus as i16 // 코어 수가 적으면 모든 코어 사용
    };

    // ONNX Runtime 환경 생성
    let environment = Environment::builder()
        .with_name("remove_background") // 환경 이름 설정
        .with_log_level(LoggingLevel::Warning) // 로그 레벨 설정
        .build()?;

    // 세션 생성 (GPU 가속은 사용 안하고 CPU 최적화)
    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)? // 그래프 최적화 
        .with_number_threads(n_threads)? // 단일 세션 내 모델의 내부연산을 병렬 처리를 위한 스레드 수
        .with_model_from_file("modnet.onnx")
        .expect("** 모델 로딩에 실패하였습니다. modnet.onnx 파일이 .exe 파일 경로에 존재하는지 확인해주세요**"); 
        

    println!("인물 배경제거 AI모델이 성공적으로 로드되었습니다!\n");
    println!("************* 작업을 시작합니다 *************");
     // 디렉토리 순회하며 배경제거 수행
    process_directory(input_dir, temp_dir, output_dir, &mut session)?;
    // process_directory(input_dir, output_dir, session);

    println!("모든 처리가 완료되었습니다. 'output' 디렉토리를 확인하세요.");
    println!("프로그램을 종료하시려면 엔터 키를 누르세요...");

    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("입력을 읽는 데 실패했습니다.");
    println!("엔터 키가 입력되었습니다. 프로그램을 종료합니다.");
    Ok(())
}



fn process_directory(
    input_dir: &Path,
    temp_dir : &Path,
    output_dir: &Path,
    session: &mut onnxruntime::session::Session<'_>,
) -> std::io::Result<()> {
    let start_time = Instant::now();

    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {

            // 파일 확장자 확인 (이미지 파일만 처리)
            if let Some(ext) = path.extension() {
                if SUPPORTED_EXTS.contains(&ext.to_str().unwrap_or("")) {

                    println!("▶️ Input : {:?}", path.display());

                    // WebP 파일 처리
                    if ext.to_str() == Some("webp") {
                        match image_decoder::convert_webp(path.to_str().unwrap()) {
                            Ok(image) => {
                                let temp_path = temp_dir.join(path.file_stem().unwrap()).with_extension("png");
                                if let Err(e) = image.save(&temp_path) {
                                    println!("webp 변환 실패: {:?}, {}", path.display(), e);
                                    continue;
                                }
                                println!("webp -> png: {:?}", temp_path.display());
                                temp_path
                            }
                            Err(e) => {
                                println!("webp 변환 오류: {:?}, {}", path.display(), e);
                                continue;
                            }
                        }
                    }else {
                        path.clone()
                    };


                    let mut output_file = output_dir.join(path.file_stem().unwrap()); // 파일 이름 가져오기 (확장자 제거)
                    output_file.set_extension("png"); // 확장자를 "png"로 설정

                    let file_start_time = Instant::now();

                    match image_processor::process_image(&path, &output_file, session) {
                        Ok(_) => {
                            let file_duration = file_start_time.elapsed();
                            println!(
                                "[✔] Done -> {:?} (time: {:.2?})\n",
                                output_file,
                                file_duration
                            );
                        }
                        Err(e) => println!("처리 중 오류 발생: {:?}, {}", path.display(), e),
                    }
                }
            }
        }
    }

    let total_duration = start_time.elapsed(); // 전체 수행 시간 측정 완료
    println!("🕒 전체 디렉토리 처리 완료! 총 소요 시간: {:.2?}", total_duration);
    Ok(())
}