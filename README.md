# Rust Background Remover

## 📌 프로젝트 개요
Rust 기반의 이미지 배경 제거 프로그램으로, ONNX Runtime을 사용하여 **MODNet** 모델을 실행합니다.  

## MODNet이란?
**MODNet** (Matting Objective Decomposition Network)은 **ZHKKKe** 팀에서 개발한 경량 딥러닝 네트워크로,  
이미지에서 **사람을 감지하고 배경을 제거** 할 수 있는 경량 네트워크입니다.  

🔗 [MODNet 공식 저장소 (GitHub)](https://github.com/ZHKKKe/MODNet)

본 프로젝트는 MODNet 공식 저장소에서 변환한 `MODNet.onnx` 모델을 사용하여 **이미지에서 배경을 제거**한 후 결과를 저장합니다.


## 🚀 사용 기술
- **Rust**
- **ONNX Runtime**
- **MODNet (ONNX 변환)**
- **Image Processing**


### 프로젝트 구조
```
background-remover/
├── modnet.onnx             # MODNet ONNX 모델 파일
├── input/                  # 원본 이미지 저장 폴더
├── output/                 # 배경 제거된 이미지 저장 폴더
├── src/
│   ├── main.rs             # 메인 실행 파일
│   ├── image_processor.rs  # 이미지 처리 모듈
│   ├── README.md           # 프로젝트 설명서
├── Cargo.toml              # Rust 프로젝트 설정 파일
```


## 🔧 설치 및 실행 방법


## 빠른 실행 

### Windows
- https://drive.google.com/file/d/1cGUtzZIeQww7iXJ14sQCSaIm6vZi4TJh/view?usp=sharing

구글 드라이브에 다운로드를 하세요.

<img src="./screenshot/0.png" width="25%">

input 디렉토리에 인물 이미지들을 넣으세요.

<img src="./screenshot/1.png" width="70%">

예제로 들어간 이미지는 스테이블 디퓨전으로 생성되었습니다

background-remover.exe 파일을 실행하세요.

<img src="./screenshot/2.png" width="80%">

작업이 완료되면 output 디렉토리를 확인하세요.



## Rust 프로젝트 빌드

```sh
git clone https://github.com/your-repo/rust-background-remover.git
cd rust-background-remover
cargo build --release
```

## Rust 프로젝트 실행

```sh
cargo run
```

## 라이선스
- 본 프로젝트는 **MIT License**를 사용합니다.
- MODNet은 **Apache 2.0** 라이선스를 사용합니다.


