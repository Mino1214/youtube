@echo off
REM 8단계 파이프라인 검증 스크립트

echo ============================================================
echo 8단계 파이프라인 검증
echo ============================================================
echo.

if exist venv (
    call venv\Scripts\activate.bat
)

python validate_pipeline.py

echo.
pause
