"""Piper 영어 음성 모델 다운로드 스크립트"""

import sys
from pathlib import Path

try:
    # piper-tts의 다운로드 기능 사용
    import subprocess
    import requests
    import json
    
    print("Piper 영어 음성 모델 다운로드 중...")
    
    # Piper 음성 모델 URL (예시)
    # 실제로는 piper-tts의 다운로드 기능을 사용해야 함
    voice_name = "en_US-lessac-medium"
    
    # 모델 저장 경로
    model_dir = Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"모델 저장 경로: {model_dir}")
    print("\n수동 다운로드 방법:")
    print("1. https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium")
    print("2. model.onnx와 model.onnx.json 파일 다운로드")
    print(f"3. {model_dir} 디렉토리에 저장")
    print("\n또는 piper 명령어 사용:")
    print(f"  piper download --voice {voice_name}")
    
except Exception as e:
    print(f"오류: {e}")

