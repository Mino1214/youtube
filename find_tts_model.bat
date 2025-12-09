@echo off
REM TTS 모델 위치 찾기 스크립트

echo ============================================================
echo 영어 TTS 모델 위치 찾기
echo ============================================================
echo.

if not exist venv (
    echo ❌ 가상환경이 없습니다.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

python find_tts_model.py

echo.
pause
