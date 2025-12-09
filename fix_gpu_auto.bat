@echo off
REM GPU 문제 자동 수리 스크립트 (PyTorch Nightly 설치)

echo ============================================================
echo GPU 문제 자동 수리 (PyTorch Nightly 설치)
echo ============================================================
echo.

if not exist venv (
    echo ❌ 가상환경이 없습니다.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/4] 현재 GPU 상태 확인...
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.version.cuda}')
    try:
        test = torch.zeros(1).cuda()
        del test
        print('✅ GPU 테스트 성공')
    except RuntimeError as e:
        if 'kernel image' in str(e).lower():
            print('❌ GPU Compute Capability 호환성 문제 발견')
            exit(1)
        else:
            raise
else:
    print('❌ CUDA를 사용할 수 없습니다.')
    exit(1)
" 2>nul
if errorlevel 1 (
    echo.
    echo [2/4] 기존 PyTorch 제거 중...
    pip uninstall -y torch torchvision torchaudio 2>nul
    echo.
    
    echo [3/4] PyTorch Nightly 설치 중 (CUDA 12.4)...
    echo 이 작업은 몇 분이 걸릴 수 있습니다...
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
    if errorlevel 1 (
        echo.
        echo ⚠️  CUDA 12.4 실패, CUDA 12.1로 시도...
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
        if errorlevel 1 (
            echo.
            echo ⚠️  CUDA 12.1 실패, CUDA 11.8로 시도...
            pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
        )
    )
    echo.
    
    echo [4/4] 설치 확인 및 GPU 테스트...
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
    try:
        test = torch.zeros(1).cuda()
        result = test * 2
        del test, result
        torch.cuda.empty_cache()
        print('✅ GPU 테스트 성공!')
        print()
        print('=' * 60)
        print('✅ GPU 문제 수리 완료!')
        print('=' * 60)
    except RuntimeError as e:
        error_str = str(e).lower()
        if 'kernel image' in error_str or 'no kernel' in error_str:
            print('❌ GPU 테스트 실패: Compute Capability 호환성 문제')
            print()
            print('추가 해결 방법:')
            print('  1. PyTorch 소스에서 빌드')
            print('  2. GPU 드라이버 업데이트')
            print('  3. 다른 CUDA 버전 시도')
            exit(1)
        else:
            print(f'❌ GPU 테스트 실패: {e}')
            exit(1)
else:
    print('❌ CUDA를 사용할 수 없습니다.')
    exit(1)
"
    if errorlevel 1 (
        echo.
        echo ❌ GPU 문제를 해결할 수 없습니다.
        echo.
        echo 수동 해결 방법:
        echo   1. check_gpu_compatibility.bat 실행
        echo   2. GPU 드라이버 업데이트
        echo   3. PyTorch 소스에서 빌드
    ) else (
        echo.
        echo ✅ GPU 문제 수리 완료!
    )
) else (
    echo.
    echo ✅ GPU가 정상 작동 중입니다.
)

echo.
pause
