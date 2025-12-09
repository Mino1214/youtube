# 모델 저장 위치 가이드

## 📁 모델 저장 위치 요약

`huggingface-cli download` 명령어로 다운로드한 모델은 다음 위치에 저장됩니다:

### 1. **LLM 모델** (Llama, DeepSeek 등)
- **기본 위치**: `models/llm/{모델명}/`
  - 예: `models/llm/llama-3.1-8b/`
- **config.yaml에서 `model_path` 지정 시**: 지정한 경로
- **local_path가 없으면**: HuggingFace 기본 캐시
  - Windows: `C:\Users\{사용자명}\.cache\huggingface\hub\models--{모델ID}/`
  - Linux/Mac: `~/.cache/huggingface/hub/models--{모델ID}/`

### 2. **Whisper 모델** (`openai/whisper-large-v3`)
- **저장 위치**: HuggingFace 기본 캐시
  - Windows: `C:\Users\{사용자명}\.cache\huggingface\hub\models--openai--whisper-large-v3/`
  - Linux/Mac: `~/.cache/huggingface/hub/models--openai--whisper-large-v3/`

### 3. **Piper TTS 모델** (`neurlang/piper-onnx-kss-korean`)
- **저장 위치**: `models/tts/{모델ID}/`
  - 예: `models/tts/neurlang_piper-onnx-kss-korean/`
  - 파일:
    - `piper-kss-korean.onnx`
    - `piper-kss-korean.onnx.json`

### 4. **Stable Diffusion XL**
- **저장 위치**: HuggingFace 기본 캐시
  - Windows: `C:\Users\{사용자명}\.cache\huggingface\hub\models--stabilityai--stable-diffusion-xl-base-1.0/`
  - Linux/Mac: `~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/`

### 5. **Stable Video Diffusion (SVD)**
- **저장 위치**: HuggingFace 기본 캐시
  - Windows: `C:\Users\{사용자명}\.cache\huggingface\hub\models--stabilityai--stable-video-diffusion-img2vid/`
  - Linux/Mac: `~/.cache/huggingface/hub/models--stabilityai--stable-video-diffusion-img2vid/`

## 🔍 모델 위치 확인 방법

### 1. 다운로드 스크립트 실행 시
다운로드 완료 후 자동으로 저장 위치가 표시됩니다:
```
✅ Whisper (openai/whisper-large-v3) 다운로드 완료!
📁 저장 위치: C:\Users\alsdh\.cache\huggingface\hub\models--openai--whisper-large-v3
```

### 2. HuggingFace 캐시 확인
```bash
# Windows
dir %USERPROFILE%\.cache\huggingface\hub

# Linux/Mac
ls ~/.cache/huggingface/hub/
```

### 3. Python으로 확인
```python
from pathlib import Path

# HuggingFace 캐시 위치
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
print(f"캐시 위치: {cache_dir}")

# 특정 모델 확인
model_id = "openai/whisper-large-v3"
model_name_safe = model_id.replace("/", "--")
model_path = cache_dir / f"models--{model_name_safe}"
print(f"모델 위치: {model_path}")
print(f"존재 여부: {model_path.exists()}")
```

## 📂 프로젝트 내 로컬 저장소

일부 모델은 프로젝트 디렉토리 내에 저장됩니다:

```
aivideo/
├── models/
│   ├── llm/
│   │   ├── llama-3.1-8b/          # LLM 모델 (local_path 지정 시)
│   │   └── deepseek-r1-7b/        # DeepSeek 모델 (local_path 지정 시)
│   └── tts/
│       └── neurlang_piper-onnx-kss-korean/  # Piper 한국어 모델
│           ├── piper-kss-korean.onnx
│           └── piper-kss-korean.onnx.json
```

## 💡 중요 사항

1. **HuggingFace 캐시**: 대부분의 모델은 HuggingFace 기본 캐시에 저장됩니다.
   - 이 위치는 `transformers`, `diffusers` 라이브러리가 자동으로 찾습니다.
   - 별도 설정 없이도 모델을 사용할 수 있습니다.

2. **로컬 경로 지정**: `config.yaml`에서 `model_path`를 지정하면 해당 경로에 저장됩니다.
   ```yaml
   llm:
     model: "llama-3.1-8b"
     model_path: "models/llm/llama-3.1-8b"  # 이 경로에 저장
   ```

3. **디스크 공간**: 
   - HuggingFace 캐시는 사용자 홈 디렉토리에 저장됩니다.
   - 모델 크기가 크므로 (총 ~50GB) 충분한 디스크 공간이 필요합니다.

4. **캐시 정리**: 필요시 HuggingFace 캐시를 정리할 수 있습니다:
   ```bash
   # 특정 모델만 삭제
   rm -rf ~/.cache/huggingface/hub/models--{모델ID}
   
   # 전체 캐시 삭제 (주의!)
   rm -rf ~/.cache/huggingface/hub/*
   ```

## 🔄 모델 재사용

한 번 다운로드한 모델은:
- 다른 프로젝트에서도 재사용 가능 (HuggingFace 캐시 공유)
- `transformers`, `diffusers` 등이 자동으로 캐시에서 로드
- 중복 다운로드 불필요

## 📝 요약

| 모델 타입 | 기본 저장 위치 | 로컬 경로 지정 가능 |
|---------|-------------|----------------|
| LLM | `models/llm/{모델명}/` 또는 캐시 | ✅ |
| Whisper | HuggingFace 캐시 | ❌ |
| Piper TTS | `models/tts/{모델ID}/` | ❌ |
| SDXL | HuggingFace 캐시 | ❌ |
| SVD | HuggingFace 캐시 | ❌ |

**대부분의 모델은 HuggingFace 캐시에 저장되며, 프로그램이 자동으로 찾아서 사용합니다!**
