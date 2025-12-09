@echo off
echo ========================================
echo Piper TTS 설치 스크립트
echo ========================================
echo.

echo [1/2] piper-tts 설치 중...
pip install piper-tts

echo.
echo [2/2] 영어 음성 모델 다운로드 중...
python -c "from piper.download import ensure_voice_exists; ensure_voice_exists('en_US-lessac-medium', [])"

echo.
echo ========================================
echo 완료!
echo ========================================
pause

