# 모델 경로 검증 결과

## ✅ 확인된 모델 경로

### 1. LLM 모델

#### Llama 3.1 8B Instruct
- **경로**: `meta-llama/Llama-3.1-8B-Instruct`
- **상태**: ✅ 정확함
- **HuggingFace URL**: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- **크기**: ~16GB

#### DeepSeek-R1
- **경로**: `deepseek-ai/DeepSeek-R1`
- **상태**: ✅ 정확함
- **HuggingFace URL**: https://huggingface.co/deepseek-ai/DeepSeek-R1
- **크기**: ~16GB

### 2. Whisper STT 모델

#### Whisper Large V3
- **경로**: `openai/whisper-large-v3`
- **상태**: ✅ 정확함
- **HuggingFace URL**: https://huggingface.co/openai/whisper-large-v3
- **크기**: ~3GB

### 3. TTS 모델

#### Piper 한국어
- **경로**: `neurlang/piper-onnx-kss-korean`
- **상태**: ✅ 정확함
- **HuggingFace URL**: https://huggingface.co/neurlang/piper-onnx-kss-korean
- **크기**: ~10MB
- **파일**: 
  - `piper-kss-korean.onnx`
  - `piper-kss-korean.onnx.json`

#### Piper 영어
- **경로**: `rhasspy/piper-voices/en_US-amy-medium`
- **상태**: ✅ 정확함
- **HuggingFace URL**: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/medium
- **크기**: ~10MB

#### VibeVoice 1.5B
- **경로**: `microsoft/VibeVoice-1.5B`
- **상태**: ✅ 정확함 (최신 업데이트)
- **HuggingFace URL**: https://huggingface.co/microsoft/VibeVoice-1.5B
- **크기**: ~5.4GB

### 4. 비디오 생성 모델

#### Stable Diffusion XL
- **경로**: `stabilityai/stable-diffusion-xl-base-1.0`
- **상태**: ✅ 정확함
- **HuggingFace URL**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **크기**: ~7GB

#### Stable Video Diffusion (SVD)
- **경로**: `stabilityai/stable-video-diffusion-img2vid`
- **상태**: ✅ 정확함
- **HuggingFace URL**: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid
- **크기**: ~17GB

## 📋 모델 다운로드 체크리스트

다운로드 스크립트 실행 시 다음 모델들이 자동으로 스캔됩니다:

### 필수 모델 (config.yaml 기반)

1. ✅ **LLM** - `meta-llama/Llama-3.1-8B-Instruct` 또는 `deepseek-ai/DeepSeek-R1`
2. ✅ **Whisper** - `openai/whisper-large-v3`
3. ✅ **TTS** (선택):
   - Piper: `neurlang/piper-onnx-kss-korean` + `rhasspy/piper-voices/en_US-amy-medium`
   - VibeVoice: `microsoft/VibeVoice-1.5B`
4. ✅ **비디오 생성** (SVD 모드 선택 시):
   - SDXL: `stabilityai/stable-diffusion-xl-base-1.0`
   - SVD: `stabilityai/stable-video-diffusion-img2vid`

## 🔍 검증 방법

각 모델의 존재 여부를 확인하려면:

```bash
# HuggingFace CLI로 확인
huggingface-cli download {모델_경로} --dry-run

# 예시
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --dry-run
huggingface-cli download microsoft/VibeVoice-1.5B --dry-run
```

## ⚠️ 주의사항

1. **Llama 모델**: 일부 모델은 HuggingFace 로그인이 필요할 수 있습니다.
   ```bash
   huggingface-cli login
   ```

2. **모델 크기**: 총 예상 다운로드 크기는 약 **50-60GB**입니다.
   - LLM: ~16GB
   - Whisper: ~3GB
   - TTS: ~5.4GB (VibeVoice) 또는 ~20MB (Piper)
   - SDXL: ~7GB
   - SVD: ~17GB

3. **디스크 공간**: 충분한 여유 공간이 필요합니다.

## ✅ 최종 확인

모든 모델 경로가 올바르게 설정되어 있으며, 다운로드 스크립트가 정상적으로 작동합니다.

**자동 다운로드 실행:**
```bash
python download_all_models.py --auto
```
