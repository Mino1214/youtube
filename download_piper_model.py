"""Piper 영어 음성 모델 자동 다운로드"""

import requests
from pathlib import Path
import sys

def download_piper_model():
    """Piper 영어 음성 모델 다운로드"""
    
    print("=" * 60)
    print("Piper 영어 음성 모델 다운로드")
    print("=" * 60)
    
    # 모델 정보
    voice_name = "en_US-lessac-medium"
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    model_path = f"en/en_US/lessac/medium"
    
    # 저장 경로
    model_dir = Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "lessac" / "medium"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = model_dir / "model.onnx"
    config_file = model_dir / "model.onnx.json"
    
    # 이미 다운로드된 경우
    if model_file.exists() and config_file.exists():
        print(f"✅ 모델이 이미 존재합니다: {model_file}")
        return True
    
    # 다운로드 URL
    model_url = f"{base_url}/{model_path}/model.onnx"
    config_url = f"{base_url}/{model_path}/model.onnx.json"
    
    try:
        print(f"\n모델 다운로드 중...")
        print(f"URL: {model_url}")
        
        # 모델 파일 다운로드
        print("model.onnx 다운로드 중...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r진행률: {percent:.1f}%", end="", flush=True)
        
        print("\n✅ model.onnx 다운로드 완료")
        
        # 설정 파일 다운로드
        print("model.onnx.json 다운로드 중...")
        response = requests.get(config_url)
        response.raise_for_status()
        
        with open(config_file, "wb") as f:
            f.write(response.content)
        
        print("✅ model.onnx.json 다운로드 완료")
        
        print("\n" + "=" * 60)
        print("✅ 모델 다운로드 완료!")
        print(f"저장 위치: {model_dir}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 다운로드 실패: {e}")
        print("\n수동 다운로드 방법:")
        print(f"1. {model_url} 방문")
        print(f"2. 파일을 {model_dir}에 저장")
        return False

if __name__ == "__main__":
    success = download_piper_model()
    sys.exit(0 if success else 1)

