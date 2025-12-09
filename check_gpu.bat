@echo off
REM GPU 확인 스크립트

echo ============================================================
echo GPU 상태 확인
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
print('PyTorch GPU 상태 확인')
print('=' * 60)
print()

print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU 개수: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
else:
    print()
    print('❌ CUDA를 사용할 수 없습니다.')
    print()
    print('가능한 원인:')
    print('  1. PyTorch가 CPU 버전으로 설치됨')
    print('  2. CUDA 드라이버가 설치되지 않음')
    print('  3. PyTorch와 CUDA 버전이 호환되지 않음')
    print()
    print('해결 방법:')
    print('  1. NVIDIA 드라이버 설치 확인: nvidia-smi')
    print('  2. PyTorch GPU 버전 재설치:')
    print('     pip install torch --index-url https://download.pytorch.org/whl/cu121')
    print('  3. 또는 config.yaml에서 force_gpu: true 설정')
"

echo.
pause
