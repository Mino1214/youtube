# 실행 가이드

## 🚀 빠른 시작

### 1단계: 필수 도구 설치

#### FFmpeg 설치 (필수)

**Windows:**
1. [FFmpeg 공식 사이트](https://www.gyan.dev/ffmpeg/builds/)에서 다운로드
2. 또는 Chocolatey 사용:
   ```powershell
   choco install ffmpeg
   ```
3. 또는 Scoop 사용:
   ```powershell
   scoop install ffmpeg
   ```

**Ubuntu:**
```bash
sudo apt update
sudo apt install ffmpeg
```

#### Python 패키지 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv

# Windows에서 활성화
venv\Scripts\activate

# Ubuntu에서 활성화
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

#### GPU 지원 (선택사항, 권장)

**Windows:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Ubuntu:**
```bash
pip install torch torchvision torchaudio
```

### 2단계: 모델 다운로드

모델은 첫 실행 시 자동으로 다운로드됩니다. 수동 다운로드도 가능합니다:

#### Whisper 모델
- 자동 다운로드 (첫 실행 시)

#### LLM 모델
HuggingFace CLI 설치:
```bash
pip install huggingface-hub
huggingface-cli login  # 필요시
```

**DeepSeek-R1 7B:**
```bash
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir models/llm/deepseek-r1-7b
```

**Llama 3.1 8B:**
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llm/llama-3.1-8b
```

#### Piper TTS 모델
- 자동 다운로드 (첫 실행 시)

### 3단계: 실행

#### 기본 실행 (로컬 파일)
```bash
python run_pipeline.py --input video.mp4 --output output.mp4
```

#### YouTube URL 변환
```bash
python run_pipeline.py --input "https://www.youtube.com/watch?v=..." --output output.mp4
```

#### 설정 파일 지정
```bash
python run_pipeline.py --input video.mp4 --output output.mp4 --config config.yaml
```

#### 상세 로그 출력
```bash
python run_pipeline.py --input video.mp4 --output output.mp4 --verbose
```

#### 임시 파일 유지 (디버깅)
```bash
python run_pipeline.py --input video.mp4 --output output.mp4 --no-cleanup
```

## 📋 실행 예제

### 예제 1: 간단한 비디오 변환
```bash
python run_pipeline.py -i test_video.mp4 -o korean_video.mp4
```

### 예제 2: YouTube 비디오 변환
```bash
python run_pipeline.py -i "https://www.youtube.com/watch?v=dQw4w9WgXcQ" -o output.mp4
```

### 예제 3: 상세 로그와 함께 실행
```bash
python run_pipeline.py -i video.mp4 -o output.mp4 -v
```

## ⚠️ 주의사항

1. **첫 실행 시 시간이 오래 걸립니다**
   - 모델 다운로드: 수 GB
   - 모델 로딩: 몇 분 소요

2. **GPU 메모리**
   - 16GB VRAM 권장
   - 부족 시 `config.yaml`에서 `batch_size` 줄이기

3. **디스크 공간**
   - 모델 저장: 약 20-30GB
   - 임시 파일: 비디오 크기에 따라 다름

## 🔧 문제 해결

### FFmpeg를 찾을 수 없음
- FFmpeg가 PATH에 추가되었는지 확인
- Windows: 시스템 환경 변수 확인
- 재시작 후 다시 시도

### 모델 다운로드 실패
- 인터넷 연결 확인
- HuggingFace 토큰 설정 (필요시)

### GPU 메모리 부족
- `config.yaml`에서 `use_gpu: false`로 설정
- 또는 `batch_size` 줄이기

### TTS 오류
- Piper 모델이 올바르게 다운로드되었는지 확인
- `models/tts/` 디렉토리 확인

