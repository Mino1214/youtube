"""CUDA 12.8 테스트 스크립트"""

import torch

print("=" * 60)
print("CUDA 12.8 테스트")
print("=" * 60)

print(f"\nPyTorch 버전: {torch.__version__}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    print("\n[테스트] CUDA 텐서 연산...")
    try:
        x = torch.randn(10, 10, device="cuda")
        y = x @ x
        print("✅ CUDA 텐서 연산 성공!")
        print(f"결과 shape: {y.shape}")
        print(f"결과 device: {y.device}")
        
        # 추가 테스트: 더 복잡한 연산
        print("\n[테스트] 복잡한 CUDA 연산...")
        a = torch.randn(100, 100, device="cuda")
        b = torch.randn(100, 100, device="cuda")
        c = torch.matmul(a, b)
        print("✅ 복잡한 CUDA 연산 성공!")
        
        print("\n" + "=" * 60)
        print("✅ CUDA 12.8 정상 작동!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ CUDA 연산 실패: {e}")
        print("\n" + "=" * 60)
        print("❌ CUDA 오류 발생")
        print("=" * 60)
else:
    print("❌ CUDA를 사용할 수 없습니다.")

