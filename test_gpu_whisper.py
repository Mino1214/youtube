"""GPU에서 Whisper 실제 테스트"""

import sys
import logging
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("GPU Whisper 테스트")
print("=" * 60)

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A'}")

# 간단한 CUDA 테스트
print("\n[1/2] 기본 CUDA 연산 테스트...")
try:
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print("✅ 기본 CUDA 연산 성공")
except Exception as e:
    print(f"❌ 기본 CUDA 연산 실패: {e}")
    sys.exit(1)

# Whisper 모델 로드 테스트
print("\n[2/2] Whisper 모델 CUDA 로드 테스트...")
try:
    import whisper
    print("Whisper 모델 로드 시도 중...")
    model = whisper.load_model("base", device="cuda")  # 작은 모델로 먼저 테스트
    print("✅ Whisper 모델 CUDA 로드 성공!")
    print(f"모델 디바이스: {next(model.parameters()).device}")
except Exception as e:
    print(f"❌ Whisper 모델 CUDA 로드 실패: {e}")
    if "no kernel image" in str(e).lower():
        print("\n⚠️ sm_120 지원 문제입니다.")
        print("해결 방법:")
        print("1. PyTorch Nightly 빌드 설치 (권장)")
        print("2. 또는 PyTorch 소스에서 직접 빌드")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ GPU 모드 정상 작동!")
print("=" * 60)

