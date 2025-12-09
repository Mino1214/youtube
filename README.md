# 로컬 AI 자동 비디오 변환 프로젝트

외국 비디오를 한국어 나레이션과 자막이 있는 비디오로 완전 자동 변환하는 로컬 파이프라인입니다.

## 🎯 기능

- **완전 자동화**: 단일 명령어로 전체 파이프라인 실행
- **로컬 전용**: 모든 처리가 로컬에서 실행 (클라우드 API 없음)
- **다중 플랫폼**: Windows 10/11 및 Ubuntu 22.04+ 지원
- **GPU 가속**: CUDA 지원 (Windows 및 Ubuntu)
- **다양한 입력**: 로컬 파일 또는 YouTube URL 지원

## 📋 요구사항

### 하드웨어
- **GPU**: NVIDIA GPU (CUDA 지원, 권장: 16GB+ VRAM)
- **CPU**: 최소 4코어 (GPU 없이도 실행 가능하나 느림)

### 소프트웨어
- **OS**: Windows 10/11 또는 Ubuntu 22.04+
- **Python**: 3.10 이상
- **FFmpeg**: 비디오/오디오 처리용
- **CUDA**: GPU 사용 시 (선택사항)

## 🚀 설치

### 1. 저장소 클론 또는 다운로드

```bash
git clone <repository-url>
cd aivideo
```

### 2. Python 환경 설정

#### Windows (Conda 권장)
```bash
conda create -n aivideo python=3.10
conda activate aivideo
```

#### Ubuntu
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 3. FFmpeg 설치

#### Windows
1. [FFmpeg 공식 사이트](https://ffmpeg.org/download.html)에서 다운로드
2. 시스템 PATH에 추가

또는 Chocolatey 사용:
```powershell
choco install ffmpeg
```

#### Ubuntu
```bash
sudo apt update
sudo apt install ffmpeg
```

### 4. Python 패키지 설치

```bash
pip install -r requirements.txt
```

#### GPU 지원 (CUDA)
PyTorch CUDA 버전 설치:
```bash
# Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Ubuntu
pip install torch torchvision torchaudio
```

### 5. 모델 다운로드

#### Whisper 모델
자동 다운로드됩니다 (첫 실행 시).

#### LLM 모델
HuggingFace에서 자동 다운로드되거나, 수동으로 다운로드 가능:

**DeepSeek-R1 7B:**
```bash
# HuggingFace CLI 사용
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir models/llm/deepseek-r1-7b
```

**Llama 3.1 8B:**
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llm/llama-3.1-8b
```

#### TTS 모델 (Piper)
자동 다운로드됩니다 (첫 실행 시).

## 📖 사용법

### 기본 사용법

```bash
python run_pipeline.py --input video.mp4 --output output.mp4
```

### YouTube URL 변환

```bash
python run_pipeline.py --input "https://www.youtube.com/watch?v=..." --output output.mp4
```

### 설정 파일 지정

```bash
python run_pipeline.py --input video.mp4 --output output.mp4 --config custom_config.yaml
```

### 상세 로그 출력

```bash
python run_pipeline.py --input video.mp4 --output output.mp4 --verbose
```

### 임시 파일 유지 (디버깅)

```bash
python run_pipeline.py --input video.mp4 --output output.mp4 --no-cleanup
```

## ⚙️ 설정 파일 (config.yaml)

주요 설정 옵션:

### LLM 모델 선택
```yaml
llm:
  model: "deepseek-r1-7b"  # 또는 "llama-3.1-8b"
  use_gpu: true
  batch_size: 4
```

### TTS 모델 선택
```yaml
tts:
  model: "piper"  # 또는 "styletts2"
  piper:
    voice: "ko_KR-hyeri-medium"
    speed: 1.0
```

### 자막 옵션
```yaml
subtitles:
  mode: "korean_only"  # 또는 "both_languages"
```

자세한 설정은 `config.yaml` 파일을 참조하세요.

## 📂 프로젝트 구조

```
aivideo/
├── models/              # 모델 저장 디렉토리
│   ├── whisper/         # Whisper 모델 캐시
│   ├── llm/             # LLM 모델
│   └── tts/             # TTS 모델
├── src/
│   ├── audio_extract.py    # 오디오 추출
│   ├── stt_whisper.py      # Whisper STT
│   ├── translate_llm.py    # LLM 번역
│   ├── tts_korean.py       # 한국어 TTS
│   ├── subtitles.py        # 자막 생성
│   ├── renderer.py         # 비디오 렌더링
│   └── pipeline.py         # 파이프라인 조율
├── config.yaml          # 설정 파일
├── run_pipeline.py      # CLI 진입점
├── requirements.txt     # Python 의존성
└── README.md            # 이 파일
```

## 🔄 파이프라인 단계

1. **오디오 추출**: FFmpeg로 비디오에서 오디오 추출
2. **STT 처리**: Whisper large-v3로 영어 음성을 텍스트로 변환
3. **번역**: 로컬 LLM으로 영어→한국어 해설 스타일 번역
4. **TTS 생성**: Piper로 한국어 나레이션 생성
5. **자막 생성**: 타임코드 기반 SRT 자막 파일 생성
6. **비디오 렌더링**: 원본 비디오(음소거) + 한국어 나레이션 + 자막 결합
7. **출력**: 최종 비디오 파일 생성

## 🐛 트러블슈팅

### FFmpeg를 찾을 수 없음
- FFmpeg가 설치되어 있고 PATH에 추가되어 있는지 확인
- Windows: 시스템 환경 변수에서 PATH 확인
- Ubuntu: `which ffmpeg` 명령어로 확인

### GPU 메모리 부족
- `config.yaml`에서 `batch_size`를 줄이기
- LLM 모델을 더 작은 모델로 변경
- CPU 모드 사용: `use_gpu: false`

### 모델 다운로드 실패
- 인터넷 연결 확인
- HuggingFace 토큰 설정 (필요시):
  ```bash
  huggingface-cli login
  ```

### TTS 음성 품질 문제
- `config.yaml`에서 TTS 파라미터 조정:
  ```yaml
  tts:
    piper:
      speed: 1.0        # 속도 조절
      noise_scale: 0.667  # 노이즈 스케일
  ```

### 자막이 표시되지 않음
- FFmpeg가 자막 필터를 지원하는지 확인
- ASS 형식으로 변경 시도 (더 나은 호환성)

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## 🤝 기여

버그 리포트 및 기능 제안은 이슈로 등록해주세요.

## 📧 문의

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요.

---

**참고**: 이 프로젝트는 완전히 로컬에서 실행되며, 어떤 클라우드 서비스도 사용하지 않습니다. 모든 모델과 처리는 사용자의 컴퓨터에서 실행됩니다.

