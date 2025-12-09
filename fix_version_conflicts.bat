@echo off
REM 버전 충돌 해결 스크립트

echo ============================================================
echo 버전 충돌 해결
echo ============================================================
echo.

REM 가상환경 확인
if not exist venv (
    echo ❌ 가상환경이 없습니다.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/4] 충돌하는 패키지 제거 중...
pip uninstall -y numpy torch transformers diffusers 2>nul
echo ✅ 제거 완료
echo.

echo [2/4] PyTorch 재설치 (NumPy 자동 포함)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    pip install torch
)
echo.

echo [3/4] NumPy 버전 확인...
python -c "import numpy; print('NumPy:', numpy.__version__)" 2>nul
if errorlevel 1 (
    pip install "numpy>=1.24.0,<2.0.0"
)
echo.

echo [4/4] Transformers & Diffusers 재설치 (호환 버전)...
pip install "transformers>=4.40.0,<5.0.0"
pip install "diffusers>=0.27.0,<0.30.0"
echo.

echo ============================================================
echo 설치 확인...
echo ============================================================
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul || echo "❌ PyTorch 오류"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)" 2>nul || echo "❌ NumPy 오류"
python -c "import transformers; print('✅ Transformers:', transformers.__version__)" 2>nul || echo "❌ Transformers 오류"
python -c "import diffusers; print('✅ Diffusers:', diffusers.__version__)" 2>nul || echo "❌ Diffusers 오류"

echo.
echo ============================================================
echo ✅ 버전 충돌 해결 완료!
echo ============================================================
echo.
pause
