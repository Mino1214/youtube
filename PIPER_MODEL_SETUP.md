# Piper TTS 모델 설정 가이드

## 현재 상태

영어 TTS를 위해 Piper 모델이 필요하지만, 현재 모델이 다운로드되지 않아 무음 오디오로 대체되고 있습니다.

## 모델 다운로드 방법

### 방법 1: HuggingFace에서 수동 다운로드 (권장)

1. https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium 방문
2. 다음 파일 다운로드:
   - `model.onnx`
   - `model.onnx.json`
3. 저장 위치:
   ```
   C:\Users\<사용자명>\.local\share\piper\voices\en\en_US\lessac\medium\
   ```
   또는
   ```
   프로젝트폴더\models\tts\
   ```

### 방법 2: piper 명령어 사용 (설치된 경우)

```bash
piper download --voice en_US-lessac-medium
```

### 방법 3: Python 스크립트로 다운로드

```python
import requests
from pathlib import Path

# 모델 URL (예시)
model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/model.onnx"
config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/model.onnx.json"

# 저장 경로
model_dir = Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "lessac" / "medium"
model_dir.mkdir(parents=True, exist_ok=True)

# 다운로드
print("모델 다운로드 중...")
response = requests.get(model_url)
with open(model_dir / "model.onnx", "wb") as f:
    f.write(response.content)

response = requests.get(config_url)
with open(model_dir / "model.onnx.json", "wb") as f:
    f.write(response.content)

print("다운로드 완료!")
```

## 모델 다운로드 후

모델을 다운로드한 후 `python main.py`를 다시 실행하면 실제 TTS가 작동합니다.

## 참고

- 현재는 모델이 없어도 무음 오디오로 대체되어 파이프라인이 계속 진행됩니다
- 실제 TTS를 사용하려면 위 방법 중 하나로 모델을 다운로드하세요

