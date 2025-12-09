@echo off
REM 가상환경 내 손상된 NumPy 강제 삭제 및 재설치

echo ============================================================
echo 가상환경 NumPy 문제 해결
echo ============================================================
echo.

REM 가상환경 확인
if not exist venv (
    echo ❌ 가상환경이 없습니다.
    echo 먼저 가상환경을 생성하세요: python -m venv venv
    pause
    exit /b 1
)

echo [1/3] 손상된 NumPy 디렉토리 삭제 중...
REM 가상환경 내 손상된 NumPy 디렉토리 찾기 및 삭제
set NUMPY_DIRS=venv\Lib\site-packages\numpy venv\Lib\site-packages\~umpy

for %%d in (%NUMPY_DIRS%) do (
    if exist %%d (
        echo   발견: %%d
        rmdir /s /q "%%d" 2>nul
        if exist %%d (
            echo   ⚠️  삭제 실패: %%d
            echo   수동으로 삭제해주세요.
        ) else (
            echo   ✅ 삭제 완료: %%d
        )
    )
)

REM .dist-info도 삭제
if exist venv\Lib\site-packages\numpy-*.dist-info (
    echo   numpy dist-info 삭제 중...
    for /d %%d in (venv\Lib\site-packages\numpy-*.dist-info) do (
        rmdir /s /q "%%d" 2>nul
    )
)

echo.
echo [2/3] NumPy 강제 설치 중...
call venv\Scripts\activate.bat

REM --ignore-installed로 기존 설치 무시하고 설치
pip install --ignore-installed --no-deps numpy==1.24.0
if errorlevel 1 (
    echo.
    echo ⚠️  --ignore-installed 실패, 다른 방법 시도...
    pip install --force-reinstall --no-deps numpy==1.24.0
)

REM 의존성 설치
pip install numpy==1.24.0

echo.
echo [3/3] 설치 확인 중...
python -c "import numpy; print('✅ NumPy 버전:', numpy.__version__)"
if errorlevel 1 (
    echo.
    echo ❌ NumPy 설치 확인 실패
    echo.
    echo 수동 해결 방법:
    echo   1. venv\Lib\site-packages\ 폴더에서 numpy, ~umpy 디렉토리 수동 삭제
    echo   2. pip install --ignore-installed numpy==1.24.0
) else (
    echo.
    echo ============================================================
    echo ✅ NumPy 설치 완료!
    echo ============================================================
)

echo.
pause
