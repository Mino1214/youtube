# 채널 프로필 사용 가이드

여러 채널을 운영할 때 각 채널의 고유한 스타일을 자동으로 적용할 수 있는 채널 프로필 시스템입니다.

## 기능 개요

- **채널별 스타일 자동 적용**: 나레이션 숏츠, 단편 스토리 등 각 채널에 맞는 스타일 자동 적용
- **번역 스타일 커스터마이징**: 채널별로 다른 톤과 문체로 번역
- **TTS 설정 자동화**: 채널별 음성 속도, 톤 자동 설정
- **비디오 스타일 자동화**: 배경색, 자막 스타일, 최대 길이 등 자동 설정
- **출력 경로 자동 관리**: 채널별로 다른 디렉토리에 자동 저장

## 사용 방법

### 1. 기본 사용 (채널 선택)

```bash
python main.py
```

실행하면 사용 가능한 채널 목록이 표시되고, 원하는 채널을 선택할 수 있습니다.

### 2. 프로그래밍 방식 사용

```python
from src.pipeline import VideoConversionPipeline

# 채널 프로필과 함께 파이프라인 초기화
pipeline = VideoConversionPipeline(
    config_path="config.yaml",
    channel="narration_shorts"  # 채널 프로필 이름
)

# 텍스트에서 비디오 생성
result = pipeline.run_from_text(
    english_text="Your English text here...",
    output_path="output.mp4",
    channel="narration_shorts"
)
```

## 사용 가능한 채널 프로필

### 1. 나레이션 숏츠 (narration_shorts)
- **용도**: 하나의 사건을 빠르게 설명하는 숏츠
- **특징**:
  - 빠른 템포 (TTS 속도: 1.15x)
  - 짧은 비디오 (최대 60초)
  - 큰 자막 (28px)
  - 뉴스 해설자 톤

### 2. 단편 스토리 (short_story)
- **용도**: 긴 스토리텔링 콘텐츠
- **특징**:
  - 느린 템포 (TTS 속도: 0.95x)
  - 긴 비디오 (최대 600초)
  - 중간 크기 자막 (24px)
  - 스토리텔러 톤

### 3. 교육 콘텐츠 (educational)
- **용도**: 교육적이고 설명적인 콘텐츠
- **특징**:
  - 보통 템포 (TTS 속도: 1.0x)
  - 중간 길이 비디오 (최대 300초)
  - 밝은 배경
  - 교육자 톤

### 4. 뉴스 리뷰 (news_review)
- **용도**: 뉴스 분석 및 리뷰
- **특징**:
  - 약간 빠른 템포 (TTS 속도: 1.1x)
  - 중간 길이 비디오 (최대 180초)
  - 전문적인 톤

### 5. 유아용 (kids)
- **용도**: 유아를 위한 교육적이고 친근한 콘텐츠
- **특징**:
  - 매우 느린 템포 (TTS 속도: 0.85x) - 아이들이 이해하기 쉽게
  - 짧은 비디오 (최대 120초) - 집중력 고려
  - 매우 큰 자막 (36px) - 읽기 학습에 도움
  - 밝은 파스텔 배경 (#fff8e1) - 아이들에게 친근한 색상
  - 간단하고 명확한 언어, 반복적인 설명
  - 친근하고 따뜻한 톤

## 채널 프로필 커스터마이징

`channels.yaml` 파일을 수정하여 채널 프로필을 커스터마이징할 수 있습니다.

### 예시: 새로운 채널 추가

```yaml
channels:
  my_custom_channel:
    name: "내 커스텀 채널"
    description: "내가 만든 커스텀 스타일"
    
    translation_style:
      tone: "스토리텔러"
      style: "친근하고 재미있는 문체"
      max_length_ratio: 1.0
      remove_fillers: false
      
    tts:
      voice: "ko_KR-hyeri-medium"
      speed: 1.05
      emotion: "friendly"
      
    video:
      max_duration: 120
      resolution: "1080p"
      fps: 30
      background_color: "#2a2a2a"
      text_style: "modern"
      
    subtitles:
      font_size: 26
      position: "bottom"
      animation: "smooth"
      highlight_keywords: false
      
    output:
      directory: "output/my_custom"
      filename_pattern: "custom_{timestamp}_{title}.mp4"
      auto_upload: false
```

## 출력 파일 관리

각 채널 프로필은 자동으로 지정된 디렉토리에 파일을 저장합니다:

- **나레이션 숏츠**: `output/narration_shorts/`
- **단편 스토리**: `output/short_story/`
- **교육 콘텐츠**: `output/educational/`
- **뉴스 리뷰**: `output/news_review/`

파일명은 `filename_pattern`에 따라 자동 생성됩니다.

## 번역 스타일

각 채널은 고유한 번역 스타일을 가집니다:

- **뉴스 해설자**: 객관적이고 사실 중심의 설명
- **스토리텔러**: 서사적이고 몰입감 있는 문체
- **교육자**: 명확하고 이해하기 쉬운 설명

번역 스타일은 `translation_style` 섹션에서 설정할 수 있습니다.

## 주의사항

1. **채널 프로필 파일**: `channels.yaml` 파일이 프로젝트 루트에 있어야 합니다.
2. **기본 채널**: 채널을 선택하지 않으면 `default_channel` 설정이 사용됩니다.
3. **설정 우선순위**: 채널 프로필 설정이 `config.yaml`의 기본 설정을 덮어씁니다.

## 문제 해결

### 채널을 찾을 수 없다는 오류
- `channels.yaml` 파일이 프로젝트 루트에 있는지 확인
- 채널 이름이 정확한지 확인 (대소문자 구분)

### 출력 디렉토리가 생성되지 않음
- 디렉토리 생성 권한 확인
- `output` 디렉토리에 쓰기 권한이 있는지 확인

### 번역 스타일이 적용되지 않음
- `translation_style` 섹션이 채널 프로필에 올바르게 설정되어 있는지 확인
- 파이프라인 초기화 시 `channel` 파라미터가 전달되었는지 확인
