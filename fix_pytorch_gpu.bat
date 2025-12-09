@echo off
REM PyTorch GPU 버전으로 교체 (의존성 유지)

echo ============================================================
echo PyTorch GPU 버전으로 교체
echo ============================================================
echo.

if not exist venv (
    echo ❌ 가상환경이 없습니다.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/4] 현재 상태 확인...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo.

echo [2/4] 기존 PyTorch 제거 중...
pip uninstall -y torch torchvision torchaudio
echo.

echo [3/4] PyTorch GPU 버전 설치 중...
echo CUDA 13.1 호환 버전 설치 시도...
echo (CUDA 12.4 버전 사용 - CUDA 13.1과 호환)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo ⚠️  CUDA 12.4 버전 실패, CUDA 12.1로 시도...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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
else:
    print()
    print('⚠️  CUDA를 사용할 수 없습니다.')
    print('   다음을 확인하세요:')
    print('   1. NVIDIA 드라이버 설치')
    print('   2. CUDA Toolkit 설치')
    print('   3. nvidia-smi 명령어로 GPU 확인')
"

echo.
pause
