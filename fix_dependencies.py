"""의존성 문제 해결 스크립트"""

import subprocess
import sys

def fix_dependencies():
    """필요한 패키지들을 최신 버전으로 업그레이드"""
    
    print("=" * 60)
    print("의존성 패키지 업그레이드")
    print("=" * 60)
    print("\n다음 패키지들을 업그레이드합니다:")
    print("  - transformers (최신 버전)")
    print("  - diffusers (최신 버전)")
    print("  - huggingface-hub (최신 버전)")
    print("  - accelerate (최신 버전)")
    print("\n⚠️  이 작업은 몇 분 걸릴 수 있습니다.\n")
    
    packages = [
        "transformers>=4.40.0",
        "diffusers>=0.27.0",
        "huggingface-hub>=0.20.0",
        "accelerate>=0.24.0",
        "safetensors>=0.4.0",
        "invisible-watermark>=0.2.0",
    ]
    
    for package in packages:
        print(f"\n업그레이드 중: {package}")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", package],
                check=True
            )
            print(f"✅ {package} 업그레이드 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 업그레이드 실패: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ 모든 패키지 업그레이드 완료!")
    print("=" * 60)
    print("\n이제 프로그램을 다시 실행해보세요.")
    
    return True

if __name__ == "__main__":
    try:
        fix_dependencies()
    except KeyboardInterrupt:
        print("\n\n작업이 취소되었습니다.")
        sys.exit(1)
