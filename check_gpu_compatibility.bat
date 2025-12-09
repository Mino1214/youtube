@echo off
REM GPU 호환성 확인 스크립트

echo ============================================================
echo GPU 호환성 확인
echo ============================================================
echo.

if not exist venv (
    echo ❌ 가상환경이 없습니다.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

python -c "
import torch
import sys

print('=' * 60)
print('GPU 호환성 진단')
print('=' * 60)
print()

if not torch.cuda.is_available():
    print('❌ CUDA를 사용할 수 없습니다.')
    print()
    print('확인 사항:')
    print('  1. NVIDIA 드라이버 설치: nvidia-smi')
    print('  2. PyTorch GPU 버전 설치 확인')
    sys.exit(1)

print(f'✅ CUDA 사용 가능')
print()

# GPU 정보
gpu_name = torch.cuda.get_device_name(0)
gpu_capability = torch.cuda.get_device_capability(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
cuda_version = torch.version.cuda
pytorch_version = torch.__version__

print(f'GPU: {gpu_name}')
print(f'Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}')
print(f'메모리: {gpu_memory:.2f} GB')
print(f'CUDA 버전: {cuda_version}')
print(f'PyTorch 버전: {pytorch_version}')
print()

# Compute Capability 확인
major, minor = gpu_capability
print('=' * 60)
print('Compute Capability 분석')
print('=' * 60)

if major >= 8:
    print(f'✅ Compute Capability {major}.{minor}: 최신 GPU (Ampere 이상)')
    print('   PyTorch가 완전히 지원합니다.')
elif major == 7:
    if minor >= 5:
        print(f'✅ Compute Capability {major}.{minor}: Turing/Ampere GPU')
        print('   PyTorch가 완전히 지원합니다.')
    else:
        print(f'⚠️  Compute Capability {major}.{minor}: Volta GPU')
        print('   대부분의 PyTorch 버전이 지원합니다.')
elif major == 6:
    print(f'⚠️  Compute Capability {major}.{minor}: Pascal GPU')
    print('   일부 최신 PyTorch 기능이 제한될 수 있습니다.')
else:
    print(f'❌ Compute Capability {major}.{minor}: 오래된 GPU')
    print('   PyTorch가 지원하지 않을 수 있습니다.')

print()
print('=' * 60)
print('GPU 테스트')
print('=' * 60)

try:
    # 간단한 연산 테스트
    test_tensor = torch.zeros(100, 100).cuda()
    result = torch.matmul(test_tensor, test_tensor)
    del test_tensor, result
    torch.cuda.empty_cache()
    print('✅ GPU 연산 테스트 성공')
except RuntimeError as e:
    error_str = str(e).lower()
    if 'kernel image' in error_str or 'no kernel' in error_str:
        print('❌ GPU 연산 테스트 실패: Compute Capability 호환성 문제')
        print()
        print('해결 방법:')
        print('  1. PyTorch Nightly 버전 설치:')
        print('     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124')
        print()
        print('  2. 또는 소스에서 빌드')
    else:
        print(f'❌ GPU 연산 테스트 실패: {e}')
    sys.exit(1)

print()
print('=' * 60)
print('✅ GPU 호환성 확인 완료!')
print('=' * 60)
"

echo.
pause
