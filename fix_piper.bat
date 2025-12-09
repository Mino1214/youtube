@echo off
REM Piper TTS 설치 및 수정

echo ============================================================
echo Piper TTS 설치
echo ============================================================
echo.

REM 가상환경 확인
if not exist venv (
    echo ❌ 가상환경이 없습니다.
    echo 먼저 가상환경을 생성하세요: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/3] NumPy 확인...
python -c "import numpy; print('✅ NumPy:', numpy.__version__)" 2>nul
if errorlevel 1 (
    echo NumPy 설치 중...
    pip install "numpy>=1.24.0,<2.0.0"
)
echo.

echo [2/3] Piper TTS 설치 중...
pip install "piper-tts>=1.2.0"
echo.

echo [3/3] 설치 확인...
python -c "from piper import PiperVoice; print('✅ Piper TTS 설치 완료')" 2>nul
if errorlevel 1 (
    echo ❌ Piper TTS 설치 실패
    echo.
    echo 수동 설치 시도:
    pip install --upgrade piper-tts
) else (
    echo.
    echo ============================================================
    echo ✅ Piper TTS 설치 완료!
    echo ============================================================
)

echo.
pause
