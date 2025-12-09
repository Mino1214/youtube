@echo off
REM 가상환경 완전 삭제 및 재생성 (호환 버전으로)

echo ============================================================
echo 가상환경 완전 재생성 (호환 버전)
echo ============================================================
echo.

echo [1/4] 기존 가상환경 삭제 중...
if exist venv (
    rmdir /s /q venv
    echo ✅ 기존 가상환경 삭제 완료
) else (
    echo 기존 가상환경 없음
)

echo.
echo [2/4] 새 가상환경 생성 중...
python -m venv venv
if errorlevel 1 (
    echo ❌ 가상환경 생성 실패
    pause
    exit /b 1
)
echo ✅ 가상환경 생성 완료

echo.
echo [3/4] pip 업그레이드...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel

echo.
echo [4/4] 호환 버전 의존성 설치 중...
echo (이 과정은 시간이 걸릴 수 있습니다...)
echo.
call install_dependencies.bat

echo.
echo ============================================================
echo ✅ 가상환경 재생성 완료!
echo ============================================================
echo.
echo 가상환경 활성화: venv\Scripts\activate
echo.
pause
