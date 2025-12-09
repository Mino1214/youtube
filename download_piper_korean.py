"""Piper 한국어 음성 모델 다운로드 스크립트"""

import requests
from pathlib import Path
import sys

def download_piper_korean(voice_name: str = "ko_KR-hyeri-medium"):
    """Piper 한국어 음성 모델 다운로드"""
    
    print("=" * 60)
    print("Piper 한국어 음성 모델 다운로드")
    print("=" * 60)
    
    # HuggingFace Piper Voices 저장소
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    
    # 음성 이름을 경로로 변환
    # HuggingFace 구조: ko/ko_KR/hyeri/medium 또는 ko/ko_KR/kss/medium
    # 실제로는 직접 확인이 필요함
    
    # 일반적인 한국어 음성들
    known_voices = {
        "ko_KR-hyeri-medium": "ko/ko_KR/hyeri/medium",
        "ko_KR-kss-medium": "ko/ko_KR/kss/medium",
        "ko_KR-narae-medium": "ko/ko_KR/narae/medium",
    }
    
    if voice_name in known_voices:
        model_path = known_voices[voice_name]
        # 경로에서 변수 추출
        parts = model_path.split("/")
        lang = parts[0]
        lang_code = parts[1]
        voice_name_part = parts[2]
        quality = parts[3]
    else:
        # 자동 파싱 시도
        parts = voice_name.split("-")
        if len(parts) >= 3:
            lang_code = parts[0]  # ko_KR
            lang = lang_code.split("_")[0]  # ko
            voice_name_part = parts[1]  # hyeri 또는 kss
            quality = parts[-1]  # medium
            
            model_path = f"{lang}/{lang_code}/{voice_name_part}/{quality}"
        else:
            print(f"❌ 잘못된 음성 이름 형식: {voice_name}")
            print("지원되는 음성:")
            for v in known_voices.keys():
                print(f"  - {v}")
            return False
    
    # 저장 경로
    model_dir = Path.home() / ".local" / "share" / "piper" / "voices" / lang / lang_code / voice_name_part / quality
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
    
    print(f"\n음성: {voice_name}")
    print(f"저장 위치: {model_dir}")
    print(f"\n모델 다운로드 중...")
    
    try:
        # 모델 파일 다운로드
        print(f"model.onnx 다운로드 중...")
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
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"\r진행률: {percent:.1f}% ({size_mb:.1f}MB / {total_mb:.1f}MB)", end="", flush=True)
        
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
        print(f"1. HuggingFace 페이지 방문:")
        print(f"   https://huggingface.co/rhasspy/piper-voices/tree/main/{model_path}")
        print(f"2. 다음 파일들을 다운로드:")
        print(f"   - model.onnx")
        print(f"   - model.onnx.json")
        print(f"3. 다음 위치에 저장:")
        print(f"   {model_dir}")
        print(f"\n또는 브라우저에서 직접 다운로드:")
        print(f"   모델: {model_url}")
        print(f"   설정: {config_url}")
        return False

if __name__ == "__main__":
    voice = sys.argv[1] if len(sys.argv) > 1 else "ko_KR-hyeri-medium"
    success = download_piper_korean(voice)
    sys.exit(0 if success else 1)

