@echo off
REM 원클릭 실행 스크립트 - 자동 진단 및 수리 후 실행

echo ============================================================
echo AI Video Generator - 원클릭 실행
echo ============================================================
echo.
echo 자동으로 모든 문제를 진단하고 수리한 후 비디오를 생성합니다.
echo.

REM 가상환경 확인 및 활성화
if exist venv (
    call venv\Scripts\activate.bat
) else (
    echo ⚠️  가상환경이 없습니다. 생성 중...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM Python 진단 및 수리 스크립트 실행
python run_with_auto_fix.py

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ❌ 자동 수리 실패
    echo ============================================================
    echo.
    echo 문제를 정확히 파악하기 위해 상세 진단을 실행하세요:
    echo.
    echo   1. DIAGNOSE_GPU.bat 실행 (GPU 문제 진단)
    echo      - GPU 정보, PyTorch 버전, 호환성 분석
    echo      - 정확한 문제 원인 파악
    echo.
    echo   2. 진단 결과에 따라 수리:
    echo      - fix_gpu_auto.bat (자동 수리)
    echo      - 또는 수동 해결
    echo.
    pause
    exit /b 1
)

echo.
pause
