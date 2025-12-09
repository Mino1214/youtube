"""필수 패키지 설치 확인 스크립트"""

import sys
import importlib

# requirements.txt에서 필요한 패키지 목록
REQUIRED_PACKAGES = {
    # Core dependencies
    "torch": "torch",
    "transformers": "transformers",
    "huggingface_hub": "huggingface-hub",
    "accelerate": "accelerate",
    "sentencepiece": "sentencepiece",
    "google.protobuf": "protobuf",
    
    # Whisper STT
    "whisper": "openai-whisper",
    
    # Video/Audio processing
    "ffmpeg": "ffmpeg-python",
    "moviepy": "moviepy",
    "yt_dlp": "yt-dlp",
    
    # Utilities
    "yaml": "pyyaml",
    "tqdm": "tqdm",
    "numpy": "numpy",
    "requests": "requests",
    "PIL": "Pillow",
}

# 선택적 패키지 (없어도 작동 가능)
OPTIONAL_PACKAGES = {
    "piper": "piper-tts",
}

def check_package(module_name, package_name):
    """패키지 설치 여부 확인"""
    try:
        importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("필수 패키지 설치 확인")
    print("=" * 60)
    
    missing_packages = []
    optional_missing = []
    
    # 필수 패키지 확인
    print("\n[필수 패키지]")
    for module_name, package_name in REQUIRED_PACKAGES.items():
        is_installed, error = check_package(module_name, package_name)
        if is_installed:
            print(f"✅ {package_name}")
        else:
            print(f"❌ {package_name} - 누락됨")
            missing_packages.append(package_name)
    
    # 선택적 패키지 확인
    print("\n[선택적 패키지]")
    for module_name, package_name in OPTIONAL_PACKAGES.items():
        is_installed, error = check_package(module_name, package_name)
        if is_installed:
            print(f"✅ {package_name} (선택적)")
        else:
            print(f"⚠️  {package_name} (선택적, 없어도 작동 가능)")
            optional_missing.append(package_name)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    
    if missing_packages:
        print(f"\n❌ 누락된 필수 패키지 ({len(missing_packages)}개):")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n설치 명령어:")
        print(f"pip install {' '.join(missing_packages)}")
        return 1
    else:
        print("\n✅ 모든 필수 패키지가 설치되어 있습니다!")
        
        if optional_missing:
            print(f"\n⚠️  선택적 패키지 ({len(optional_missing)}개)가 설치되지 않았습니다:")
            for pkg in optional_missing:
                print(f"   - {pkg}")
            print("(선택적 패키지는 없어도 작동 가능합니다)")
        
        return 0

if __name__ == "__main__":
    sys.exit(main())

