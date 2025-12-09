@echo off
REM PyTorch GPU 버전 설치 (CUDA 13.1 호환)

echo ============================================================
echo PyTorch GPU 버전 설치 (CUDA 13.1)
echo ============================================================
echo.

if not exist venv (
    echo ❌ 가상환경이 없습니다.
    echo 먼저 가상환경을 생성하세요: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/4] 현재 상태 확인...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>nul || echo "PyTorch가 설치되지 않았습니다."
echo.

echo [2/4] 기존 PyTorch 제거 중...
pip uninstall -y torch torchvision torchaudio 2>nul
echo.

echo [3/4] PyTorch GPU 버전 설치 중 (CUDA 13.1 호환)...
echo.
echo CUDA 13.1은 CUDA 12.4와 호환됩니다.
echo CUDA 12.4 버전 설치 시도...
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo.
    echo ⚠️  CUDA 12.4 버전 실패, CUDA 12.1로 시도...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo.
        echo ⚠️  CUDA 12.1 버전도 실패, 최신 버전으로 시도...
        pip install torch torchvision torchaudio
    )
)
echo.

echo [4/4] 설치 확인...
python -c "
import torch
print()
print('=' * 60)
print('설치 결과')
print('=' * 60)
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print()
    print('✅ GPU 버전 설치 완료!')
    print()
    print('참고: CUDA 13.1과 PyTorch CUDA 12.4는 호환됩니다.')
else:
    print()
    print('⚠️  CUDA를 사용할 수 없습니다.')
    print('   다음을 확인하세요:')
    print('   1. NVIDIA 드라이버 설치')
    print('   2. CUDA Toolkit 13.1 설치 확인')
    print('   3. nvidia-smi 명령어로 GPU 확인')
"

echo.
pause
