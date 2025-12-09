"""모든 의존성 문제 해결 스크립트"""

import sys
import subprocess
from pathlib import Path

def fix_all():
    """모든 의존성 문제 해결"""
    print("=" * 60)
    print("모든 의존성 문제 해결")
    print("=" * 60)
    print()
    
    packages_to_fix = [
        ("torch", "torch"),  # PyTorch 먼저 설치
        ("transformers", "transformers>=4.40.0"),
        ("diffusers", "diffusers>=0.27.0"),
        ("huggingface-hub", "huggingface-hub>=0.20.0"),
    ]
    
    print("[1/5] 패키지 업그레이드 중...")
    for package_name, package_spec in packages_to_fix:
        print(f"\n  - {package_spec} 업그레이드 중...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", package_spec],
                check=True,
                capture_output=True
            )
            print(f"  ✅ {package_name} 업그레이드 완료")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  {package_name} 업그레이드 실패 (계속 진행)")
    
    print("\n[2/5] NumPy 재설치 중...")
    try:
        # 손상된 NumPy 강제 재설치
        print("  손상된 NumPy 강제 재설치 중...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "numpy==1.24.0"],
            check=True
        )
        # 의존성 설치
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "numpy==1.24.0"],
            check=True
        )
        print("✅ NumPy 재설치 완료")
    except Exception as e:
        print(f"⚠️  NumPy 재설치 실패: {e}")
        print("  수동 해결: pip install --force-reinstall --no-deps numpy==1.24.0")
    
    print("\n[3/5] 영어 TTS 모델 다운로드 중...")
    try:
        from download_english_tts import download_english_tts
        if download_english_tts():
            print("✅ 영어 TTS 모델 다운로드 완료")
        else:
            print("⚠️  영어 TTS 모델 다운로드 실패 (수동 다운로드 필요)")
    except Exception as e:
        print(f"⚠️  영어 TTS 모델 다운로드 실패: {e}")
        print("수동 다운로드: python download_english_tts.py")
    
    print("\n[4/5] 호환성 확인 중...")
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError:
        print("❌ NumPy를 import할 수 없습니다.")
        print("  수동 설치: pip install --force-reinstall --no-deps numpy==1.24.0")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # NumPy와 PyTorch 호환성 테스트
        try:
            import numpy as np
            arr = np.array([1, 2, 3])
            tensor = torch.from_numpy(arr)
            print("✅ torch.from_numpy() 테스트 성공")
        except Exception as e:
            print(f"⚠️  NumPy-PyTorch 호환성 문제: {e}")
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        print("  설치: pip install torch")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
        
        # Transformers 버전 확인
        version_parts = transformers.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        if major < 4 or (major == 4 and minor < 40):
            print("⚠️  Transformers 버전이 여전히 낮습니다. 수동 업그레이드 필요:")
            print("   pip install --upgrade transformers>=4.40.0")
        else:
            print("✅ Transformers 버전 확인 완료")
    except ImportError:
        print("❌ Transformers가 설치되지 않았습니다.")
        print("  설치: pip install transformers>=4.40.0")
            
    except Exception as e:
        print(f"⚠️  호환성 확인 실패: {e}")
    
    print("\n[5/5] 최종 확인...")
    print("\n" + "=" * 60)
    print("✅ 의존성 문제 해결 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("  1. 영어 TTS 모델이 없으면: python download_english_tts.py")
    print("  2. 프로그램 실행: python main.py")
    print()

if __name__ == "__main__":
    try:
        fix_all()
    except KeyboardInterrupt:
        print("\n\n작업이 취소되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
