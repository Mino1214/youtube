@echo off
REM PyTorch Nightly 버전 설치 (최신 GPU 지원)

echo ============================================================
echo PyTorch Nightly 버전 설치 (최신 GPU 지원)
echo ============================================================
echo.

if not exist venv (
    echo ❌ 가상환경이 없습니다.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/3] 현재 PyTorch 버전 확인...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>nul || echo "PyTorch가 설치되지 않았습니다."
echo.

echo [2/3] 기존 PyTorch 제거 중...
pip uninstall -y torch torchvision torchaudio 2>nul
echo.

echo [3/3] PyTorch Nightly 버전 설치 중 (CUDA 12.4)...
echo 이 버전은 최신 GPU compute capability를 지원합니다.
echo.
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
if errorlevel 1 (
    echo.
    echo ⚠️  CUDA 12.4 Nightly 실패, CUDA 12.1 Nightly로 시도...
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
)
echo.

echo ============================================================
echo 설치 확인...
echo ============================================================
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
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
    print()
    # GPU 테스트
    try:
        test = torch.zeros(1).cuda()
        del test
        print('✅ GPU 테스트 성공!')
    except RuntimeError as e:
        print(f'❌ GPU 테스트 실패: {e}')
else:
    print()
    print('⚠️  CUDA를 사용할 수 없습니다.')
"

echo.
pause
