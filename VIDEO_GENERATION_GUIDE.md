# 비디오 생성 가이드 (VEO3 수준)

이 프로젝트는 **Stable Diffusion XL**과 **Stable Video Diffusion (SVD)**를 사용하여 텍스트에서 고품질 비디오를 생성합니다.

## 🎬 지원하는 비디오 생성 모델

### 1. Stable Video Diffusion (SVD) - 기본 추천
- **방식**: 텍스트 → 이미지 (SDXL) → 비디오 (SVD)
- **품질**: ⭐⭐⭐⭐⭐ (최고 품질)
- **해상도**: 1024x576 (16:9) 또는 1024x1024
- **프레임 수**: 최대 25프레임 (약 1초 @ 24fps)
- **모델 크기**: 약 24GB (SDXL 7GB + SVD 17GB)

### 2. AnimateDiff
- **방식**: 텍스트 → 직접 비디오
- **품질**: ⭐⭐⭐⭐
- **해상도**: 1024x576
- **프레임 수**: 최대 16프레임

### 3. Simple Slideshow (Fallback)
- **방식**: 텍스트 슬라이드쇼
- **품질**: ⭐⭐
- **용도**: 모델 로드 실패 시 대체

## 📦 설치

### 필수 패키지 설치

```bash
pip install -r requirements.txt
```

### 추가 설치 (SDXL용)

SDXL은 다음 패키지가 필요합니다:

```bash
pip install invisible-watermark transformers accelerate safetensors
```

## ⚙️ 설정 (config.yaml)

```yaml
video_generation:
  # 모델 선택: "svd" (추천), "animatediff", "simple"
  model: "svd"
  
  # GPU 사용
  use_gpu: true
  
  # 해상도 (SDXL 권장: 1024x1024, 1024x768, 1024x576)
  width: 1024
  height: 576
  
  # 프레임레이트
  fps: 24
```

## 🚀 사용법

### 기본 사용

```bash
python main.py
```

또는

```python
from src.pipeline import VideoConversionPipeline

pipeline = VideoConversionPipeline()
result = pipeline.run_from_text(
    english_text="A beautiful sunset over the ocean",
    output_path="output.mp4"
)
```

## 💾 모델 다운로드

### Stable Diffusion XL (자동 다운로드)
- 첫 실행 시 자동으로 다운로드됩니다
- 크기: 약 7GB
- 위치: `~/.cache/huggingface/hub/`

### Stable Video Diffusion (자동 다운로드)
- SVD 모드 사용 시 자동으로 다운로드됩니다
- 크기: 약 17GB
- 위치: `~/.cache/huggingface/hub/`

### 수동 다운로드

```bash
# SDXL
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir models/sdxl

# SVD
huggingface-cli download stabilityai/stable-video-diffusion-img2vid --local-dir models/svd
```

그리고 `config.yaml`에서:

```yaml
video_generation:
  model_path: "models/sdxl"  # 또는 "models/svd"
```

## 🎯 워크플로우

### SVD 모드 (기본)

```
영어 텍스트 입력
    ↓
[SDXL] 텍스트 → 고품질 이미지 생성
    ↓
[SVD] 이미지 → 비디오 생성 (움직임 추가)
    ↓
[FFmpeg] 오디오 결합
    ↓
최종 비디오 출력
```

### AnimateDiff 모드

```
영어 텍스트 입력
    ↓
[AnimateDiff] 텍스트 → 직접 비디오 생성
    ↓
[FFmpeg] 오디오 결합
    ↓
최종 비디오 출력
```

## 💡 팁

### 1. 메모리 부족 시

```yaml
video_generation:
  use_gpu: true
  # CPU 오프로딩 자동 활성화됨
```

또는 해상도 낮추기:

```yaml
video_generation:
  width: 768
  height: 432
```

### 2. 더 긴 비디오 생성

SVD는 기본적으로 최대 25프레임(약 1초)을 생성합니다. 더 긴 비디오가 필요하면:

1. 여러 세그먼트로 나누어 생성
2. FFmpeg로 세그먼트 연결

### 3. 품질 향상

```yaml
video_generation:
  # SDXL은 더 많은 스텝으로 더 나은 품질
  # (코드에서 num_inference_steps=30으로 설정됨)
```

## ⚠️ 주의사항

1. **GPU 메모리**: SDXL + SVD는 최소 16GB VRAM 권장
2. **디스크 공간**: 모델 다운로드에 약 24GB 필요
3. **생성 시간**: 첫 생성은 모델 로드로 인해 오래 걸릴 수 있음 (5-10분)
4. **인터넷 연결**: 첫 실행 시 모델 다운로드 필요

## 🔧 트러블슈팅

### 모델 로드 실패

```bash
# HuggingFace 로그인 (필요시)
huggingface-cli login
```

### 메모리 부족

```yaml
video_generation:
  width: 768  # 해상도 낮추기
  height: 432
```

### 생성 속도가 느림

- GPU 사용 확인: `use_gpu: true`
- xformers 설치 (GPU만): `pip install xformers`

## 📚 참고 자료

- [Stable Diffusion XL 문서](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Stable Video Diffusion 문서](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
- [Diffusers 라이브러리](https://github.com/huggingface/diffusers)
