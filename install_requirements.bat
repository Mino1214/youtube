@echo off
REM 필수 패키지 설치 스크립트 (Windows)

echo ============================================================
echo 필수 패키지 설치
echo ============================================================
echo.

echo [1/3] 기본 패키지 설치 중...
pip install -r requirements.txt

echo.
echo [2/3] transformers 및 관련 패키지 설치 중...
pip install transformers>=4.40.0
pip install huggingface-hub>=0.20.0

echo.
echo [3/3] Whisper 및 TTS 관련 패키지 설치 중...
pip install librosa>=0.10.0
pip install soundfile>=0.12.0

echo.
echo ============================================================
echo 설치 완료!
echo ============================================================
echo.
echo 이제 다음 명령어로 모델을 다운로드할 수 있습니다:
echo   python download_all_models.py
echo.

pause
