mod image_processor;
use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel};
use std::error::Error;
use std::{fs, io};
use std::path::{Path};
use std::time::Instant;
use num_cpus;

fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", "*".repeat(90));
    println!("[이미지 배경 제거 프로그램]");
    println!("- 제작자 : RGB코딩");
    println!("- Version : 1.0.1\n");
    println!("[프로그램 소개]");
    println!("input 디렉토리에 있는 .jpg .png 이미지 파일들의 배경 제거하여 output 디렉토리에 저장합니다.");
    println!("! 인물 이미지만 잘 작동합니다\n");
    println!("* 공유는 자유지만, 원작자 표기는 부탁드립니다 ");
    println!("👉 https://www.youtube.com/@rgbitcode");
    println!("{}\n", "*".repeat(90));

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
        

    println!("모델이 성공적으로 로드되었습니다!\n");
    
    println!("************* 작업을 시작합니다 *************");
     // 디렉토리 순회하며 배경제거 수행
    process_directory(input_dir, output_dir, &mut session)?;
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
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    let output_file = output_dir.join(path.file_name().unwrap());
                    let file_start_time = Instant::now();

                    match image_processor::process_image(&path, &output_file, session) {
                        Ok(_) => {
                            let file_duration = file_start_time.elapsed();
                            println!(
                                "✅ Done -> {:?} (time: {:.2?})\n",
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

/*
fn process_directory(
    input_dir: &Path,
    output_dir: &Path,
    session: Arc<Session>, // 세션을 Arc로 감싸서 공유
) -> std::io::Result<()> {
    let start_time = Instant::now();

    // 파일 목록을 Vec<PathBuf>로 수집
    let files: Vec<PathBuf> = fs::read_dir(input_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension()
                    .map(|ext| ext == "jpg" || ext == "jpeg" || ext == "png")
                    .unwrap_or(false)
        })
        .collect();

    // 병렬 처리 : 수만장 이상 인물 이미지 처리하기 위해
    // TODO : 멀티쓰레드로 ONNX 세션 풀을 미리 생성하고 꺼내서 사용하도록
    files.par_iter().for_each(|path| {
        let thread_id = thread::current().id(); // 현재 쓰레드 ID 가져오기
        let output_file = output_dir.join(path.file_name().unwrap());
        let file_start_time = Instant::now();

        // 각 파일 처리
        match image_processor::process_image(path, &output_file, &mut *session) {
            Ok(_) => {
                let file_duration = file_start_time.elapsed();
                println!(
                    "[Thread {:?}] ✅ 처리 완료: {:?} -> {:?} (소요 시간: {:.2?})",
                    thread_id,
                    path.display(),
                    output_file,
                    file_duration
                );
            }
            Err(e) => println!(
                "[Thread {:?}] 처리 중 오류 발생: {:?}, {}",
                thread_id,
                path.display(),
                e
            ),
        }
    });

    let total_duration = start_time.elapsed();
    println!("🕒 전체 디렉토리 처리 완료! 총 소요 시간: {:.2?}", total_duration);

    Ok(())
}
*/