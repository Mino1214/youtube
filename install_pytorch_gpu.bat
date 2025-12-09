@echo off
REM PyTorch GPU 버전 설치 스크립트

echo ============================================================
echo PyTorch GPU 버전 설치
echo ============================================================
echo.

if not exist venv (
    echo ❌ 가상환경이 없습니다.
    echo 먼저 가상환경을 생성하세요: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/3] 현재 PyTorch 버전 확인...
python -c "import torch; print(f'현재 PyTorch: {torch.__version__}'); print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
echo.

echo [2/3] PyTorch CPU 버전 제거 중...
pip uninstall -y torch torchvision torchaudio
echo.

echo [3/3] PyTorch GPU 버전 설치 중...
echo CUDA 13.1 호환 버전 설치 시도...
echo (CUDA 12.4 버전 사용 - CUDA 13.1과 호환)
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo ⚠️  CUDA 12.4 버전 실패, CUDA 12.1로 시도...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)
echo.

echo ============================================================
echo 설치 확인...
echo ============================================================
python -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU 개수: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print()
    print('✅ PyTorch GPU 버전 설치 완료!')
else:
    print()
    print('⚠️  CUDA를 사용할 수 없습니다.')
    print('   NVIDIA 드라이버와 CUDA가 설치되어 있는지 확인하세요.')
"

echo.
pause
