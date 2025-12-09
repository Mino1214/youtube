"""NumPy 문제 해결 스크립트"""

import sys
import subprocess

def fix_numpy():
    """NumPy 재설치 및 호환성 확인"""
    print("=" * 60)
    print("NumPy 문제 해결")
    print("=" * 60)
    print()
    
    # 1. NumPy 제거
    print("[1/4] 기존 NumPy 제거 중...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], 
                      check=False, capture_output=True)
        print("✅ 완료")
    except Exception as e:
        print(f"⚠️  제거 중 오류 (무시 가능): {e}")
    
    # 2. NumPy 재설치
    print("\n[2/4] NumPy 재설치 중...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "numpy>=1.24.0"], 
                      check=True)
        print("✅ 완료")
    except Exception as e:
        print(f"❌ 설치 실패: {e}")
        return False
    
    # 3. PyTorch와 호환성 확인
    print("\n[3/4] PyTorch와 호환성 확인 중...")
    try:
        import numpy as np
        import torch
        print(f"✅ NumPy 버전: {np.__version__}")
        print(f"✅ PyTorch 버전: {torch.__version__}")
        
        # 테스트
        arr = np.array([1, 2, 3])
        tensor = torch.from_numpy(arr)
        print("✅ torch.from_numpy() 테스트 성공")
    except Exception as e:
        print(f"❌ 호환성 문제: {e}")
        print("\n추가 해결 방법:")
        print("  1. PyTorch 재설치:")
        print("     pip uninstall torch -y")
        print("     pip install torch")
        print("  2. 또는 NumPy 다운그레이드:")
        print("     pip install numpy==1.24.0")
        return False
    
    # 4. 최종 확인
    print("\n[4/4] 최종 확인 중...")
    try:
        import numpy
        import torch
        from numpy import array
        test_array = array([1, 2, 3])
        test_tensor = torch.from_numpy(test_array)
        print("✅ 모든 테스트 통과!")
        print("\n" + "=" * 60)
        print("✅ NumPy 문제 해결 완료!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"❌ 최종 확인 실패: {e}")
        print("\n수동 해결 방법:")
        print("  pip uninstall numpy torch -y")
        print("  pip install numpy==1.24.0")
        print("  pip install torch")
        return False

if __name__ == "__main__":
    try:
        success = fix_numpy()
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
