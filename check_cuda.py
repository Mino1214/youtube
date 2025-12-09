"""CUDA 호환성 확인 스크립트"""

import torch

print("=" * 60)
print("CUDA 호환성 확인")
print("=" * 60)

print(f"\nPyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
    print(f"\nGPU 정보:")
    print(f"  이름: {torch.cuda.get_device_name(0)}")
    print(f"  메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # 간단한 테스트
    try:
        print("\nCUDA 테스트 중...")
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.matmul(x, y)
        print("✅ CUDA 작동 정상!")
    except Exception as e:
        print(f"❌ CUDA 오류 발생: {e}")
        print("\n해결 방법:")
        print("1. PyTorch를 최신 버전으로 재설치")
        print("2. 또는 CPU 모드로 실행 (config.yaml에서 use_gpu: false)")
else:
    print("❌ CUDA를 사용할 수 없습니다.")

print("\n" + "=" * 60)

