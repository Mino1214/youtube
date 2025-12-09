# 모델 다운로드 가이드

## 🚀 빠른 시작

### 자동 다운로드 (추천)

```bash
python download_all_models.py
```

이 스크립트가:
1. ✅ 현재 설치된 모델 확인
2. ❌ 없는 모델 목록 표시
3. 📥 선택한 모델 순차적으로 다운로드

## 📋 필요한 모델 목록

프로젝트에서 사용하는 모든 모델:

### 1. LLM 모델 (번역용)
- **Llama 3.1 8B**: ~16GB
- **DeepSeek-R1 7B**: ~16GB (선택사항)
- **위치**: `models/llm/` 또는 HuggingFace 캐시

### 2. Whisper 모델 (STT용)
- **large-v3**: ~3GB
- **위치**: Whisper 자동 캐시 (`~/.cache/whisper/`)

### 3. TTS 모델 (음성 생성용)

#### Piper (경량, 빠름)
- **한국어**: `ko_KR-hyeri-medium` (~10MB)
- **영어**: `en_US-amy-medium` (~10MB)
- **위치**: `~/.local/share/piper/voices/`

#### VibeVoice-7B (고품질, 느림)
- **VibeVoice-7B**: ~14GB
- **위치**: HuggingFace 캐시

### 4. 비디오 생성 모델 (VEO3 수준)

#### Stable Diffusion XL (이미지 생성)
- **SDXL Base**: ~7GB
- **위치**: HuggingFace 캐시

#### Stable Video Diffusion (비디오 생성)
- **SVD**: ~17GB
- **위치**: HuggingFace 캐시

## 💾 총 다운로드 크기

**최소 구성** (Piper TTS 사용):
- LLM: ~16GB
- Whisper: ~3GB
- Piper: ~20MB
- SDXL: ~7GB
- SVD: ~17GB
- **총합: ~43GB**

**고품질 구성** (VibeVoice 사용):
- LLM: ~16GB
- Whisper: ~3GB
- VibeVoice: ~14GB
- SDXL: ~7GB
- SVD: ~17GB
- **총합: ~57GB**

## 📥 다운로드 방법

### 방법 1: 통합 스크립트 (추천)

```bash
python download_all_models.py
```

**기능:**
- ✅ 자동으로 없는 모델 감지
- ✅ 모델별 선택 다운로드 가능
- ✅ 순차적 다운로드 (중단 가능)
- ✅ 진행 상황 표시

### 방법 2: 개별 다운로드

#### LLM 모델

```bash
# Llama 3.1 8B
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llm/llama-3.1-8b

# DeepSeek-R1 7B
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir models/llm/deepseek-r1-7b
```

#### Whisper 모델

```python
import whisper
model = whisper.load_model("large-v3")  # 자동 다운로드
```

또는:

```bash
# 수동 다운로드
wget https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeef8efb14633bd4a8546ed/large-v3.pt
# ~/.cache/whisper/ 디렉토리에 저장
```

#### Piper TTS

```bash
# 한국어
python -m piper.download --voice ko_KR-hyeri-medium

# 영어
python -m piper.download --voice en_US-amy-medium
```

또는:

```bash
piper download --voice ko_KR-hyeri-medium
piper download --voice en_US-amy-medium
```

#### VibeVoice-7B

```bash
huggingface-cli download microsoft/VibeVoice-7B-hf
```

#### Stable Diffusion XL

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0
```

#### Stable Video Diffusion

```bash
huggingface-cli download stabilityai/stable-video-diffusion-img2vid
```

## ⚙️ 설정 파일 연동

다운로드한 모델을 `config.yaml`에서 지정할 수 있습니다:

```yaml
llm:
  model: "llama-3.1-8b"
  model_path: "models/llm/llama-3.1-8b"  # 로컬 경로 지정

tts:
  model: "piper"
  piper:
    voice: "ko_KR-hyeri-medium"

video_generation:
  model: "svd"
  model_path: null  # null이면 HuggingFace 캐시에서 자동 찾음
```

## 🔍 모델 확인

### 현재 설치된 모델 확인

```bash
python download_all_models.py
```

스크립트가 자동으로 모든 모델의 설치 여부를 확인합니다.

### 수동 확인

```python
# Python에서 확인
import whisper
print(whisper.available_models())  # Whisper 모델 목록

from huggingface_hub import list_models
# HuggingFace 캐시 확인
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print(os.listdir(cache_dir))
```

## ⚠️ 주의사항

1. **디스크 공간**: 최소 50GB 이상의 여유 공간 필요
2. **인터넷 속도**: 모델 크기가 크므로 안정적인 인터넷 연결 필요
3. **중단 가능**: 다운로드 중단되어도 재개 가능 (HuggingFace CLI 지원)
4. **HuggingFace 로그인**: 일부 모델은 로그인 필요할 수 있음

```bash
huggingface-cli login
```

## 🐛 트러블슈팅

### 모델을 찾을 수 없음

1. HuggingFace 캐시 확인:
   ```bash
   ls ~/.cache/huggingface/hub/
   ```

2. 로컬 경로 확인:
   ```bash
   ls models/llm/
   ```

3. `config.yaml`에서 `model_path` 설정 확인

### 다운로드 실패

1. **인터넷 연결 확인**
2. **HuggingFace 로그인**:
   ```bash
   huggingface-cli login
   ```
3. **권한 확인**: 쓰기 권한이 있는지 확인
4. **디스크 공간 확인**: 충분한 공간이 있는지 확인

### 다운로드 속도가 느림

- HuggingFace Mirror 사용 (중국 등):
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

## 📚 추가 자료

- [HuggingFace Hub 문서](https://huggingface.co/docs/hub)
- [Whisper 모델](https://github.com/openai/whisper)
- [Piper TTS](https://github.com/rhasspy/piper)
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
