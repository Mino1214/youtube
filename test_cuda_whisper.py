"""Whisper CUDA 실제 작동 테스트"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import whisper
from src.stt_whisper import create_whisper_stt
import yaml

logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("Whisper CUDA 실제 작동 테스트")
print("=" * 60)

print(f"\nPyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

print("\n[테스트 1] 간단한 CUDA 연산 테스트...")
try:
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 10).cuda()
    z = torch.matmul(x, y)
    print("✅ 기본 CUDA 연산 성공")
except Exception as e:
    print(f"❌ 기본 CUDA 연산 실패: {e}")
    sys.exit(1)

print("\n[테스트 2] Whisper 모델 로드 테스트...")
try:
    config = yaml.safe_load(open('config.yaml'))
    stt = create_whisper_stt(config)
    
    print(f"디바이스: {stt.device}")
    
    if stt.device == "cuda":
        print("Whisper 모델 로드 시도 중...")
        stt.load_model()
        print("✅ Whisper 모델 CUDA 로드 성공!")
    else:
        print("⚠️ CPU 모드로 전환됨")
        
except Exception as e:
    print(f"❌ Whisper 모델 로드 실패: {e}")
    print("\n해결 방법:")
    print("1. config.yaml에서 device를 'cpu'로 변경")
    print("2. 또는 PyTorch Nightly 빌드 사용")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

