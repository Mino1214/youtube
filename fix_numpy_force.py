"""NumPy 강제 재설치 스크립트 (손상된 패키지용)"""

import sys
import subprocess
from pathlib import Path

def fix_numpy_force():
    """손상된 NumPy 강제 재설치"""
    print("=" * 60)
    print("NumPy 강제 재설치 (손상된 패키지 복구)")
    print("=" * 60)
    print()
    
    # 1. 손상된 NumPy 디렉토리 확인 및 삭제
    print("[1/3] 손상된 NumPy 패키지 확인 중...")
    site_packages = [
        Path(sys.executable).parent / "Lib" / "site-packages",
        Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages",
    ]
    
    for site_pkg in site_packages:
        if site_pkg.exists():
            numpy_dirs = [
                site_pkg / "numpy",
                site_pkg / "~umpy",  # 손상된 디렉토리
            ]
            for numpy_dir in numpy_dirs:
                if numpy_dir.exists():
                    print(f"  발견: {numpy_dir}")
                    try:
                        import shutil
                        shutil.rmtree(numpy_dir)
                        print(f"  ✅ 삭제 완료: {numpy_dir}")
                    except Exception as e:
                        print(f"  ⚠️  삭제 실패: {e}")
    
    # 2. NumPy 강제 재설치
    print("\n[2/3] NumPy 강제 재설치 중...")
    try:
        # --force-reinstall --no-deps로 강제 설치
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "numpy==1.24.0"],
            check=True
        )
        print("✅ NumPy 강제 설치 완료")
        
        # 의존성 설치
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "numpy==1.24.0"],
            check=True
        )
        print("✅ NumPy 의존성 설치 완료")
    except Exception as e:
        print(f"❌ 설치 실패: {e}")
        print("\n수동 해결 방법:")
        print("  1. 손상된 디렉토리 수동 삭제:")
        print("     - C:\\Users\\alsdh\\AppData\\Roaming\\Python\\Python311\\site-packages\\~umpy")
        print("     - C:\\Python311\\Lib\\site-packages\\numpy (있다면)")
        print("  2. 강제 재설치:")
        print("     pip install --force-reinstall --no-deps numpy==1.24.0")
        print("     pip install numpy==1.24.0")
        return False
    
    # 3. 확인
    print("\n[3/3] 설치 확인 중...")
    try:
        import numpy as np
        print(f"✅ NumPy 버전: {np.__version__}")
        
        # 간단한 테스트
        arr = np.array([1, 2, 3])
        print(f"✅ NumPy 작동 확인: {arr}")
        
        # PyTorch와 호환성 테스트
        try:
            import torch
            tensor = torch.from_numpy(arr)
            print("✅ PyTorch와 호환성 확인 완료")
        except ImportError:
            print("⚠️  PyTorch가 없습니다. (NumPy는 정상)")
        except Exception as e:
            print(f"⚠️  PyTorch 호환성 문제: {e}")
        
        print("\n" + "=" * 60)
        print("✅ NumPy 재설치 완료!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"❌ 확인 실패: {e}")
        return False

if __name__ == "__main__":
    try:
        success = fix_numpy_force()
        if success:
            print("\n이제 프로그램을 다시 실행해보세요:")
            print("  python main.py")
        else:
            print("\n수동으로 해결이 필요합니다.")
    except KeyboardInterrupt:
        print("\n\n작업이 취소되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
