@echo off
REM 자동 Python 버전 감지 및 가상환경 설정 스크립트 (Windows)

echo ============================================================
echo 가상환경 자동 설정
echo ============================================================
echo.

REM Python 버전 확인
echo [1/6] Python 버전 확인 중...
python --version
if errorlevel 1 (
    echo.
    echo ❌ Python을 찾을 수 없습니다.
    echo Python을 설치해주세요: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Python 3.10 시도
echo.
echo [2/6] Python 3.10 확인 중...
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Python 3.10을 찾을 수 없습니다.
    echo 현재 Python 버전을 사용합니다.
    set PYTHON_CMD=python
    set PYTHON_VER=현재버전
) else (
    echo ✅ Python 3.10 발견
    set PYTHON_CMD=py -3.10
    set PYTHON_VER=3.10
)

REM 기존 가상환경 삭제 (선택사항)
if exist venv (
    echo.
    echo [3/6] 기존 가상환경 삭제 중...
    rmdir /s /q venv
    echo ✅ 기존 가상환경 삭제 완료
) else (
    echo.
    echo [3/6] 기존 가상환경 없음
)

REM 가상환경 생성
echo.
echo [4/6] 가상환경 생성 중 (Python %PYTHON_VER%)...
%PYTHON_CMD% -m venv venv
if errorlevel 1 (
    echo.
    echo ❌ 가상환경 생성 실패
    pause
    exit /b 1
)
echo ✅ 가상환경 생성 완료

REM 가상환경 활성화 및 pip 업그레이드
echo.
echo [5/6] pip 업그레이드 중...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo ✅ pip 업그레이드 완료

REM 필수 패키지 설치
echo.
echo [6/6] 필수 패키지 설치 중...
echo.
echo NumPy 설치 중...
pip install numpy==1.24.0
echo.
echo PyTorch 설치 중...
pip install torch
echo.
echo 기타 필수 패키지 설치 중...
pip install -r requirements.txt
echo.

echo ============================================================
echo ✅ 가상환경 설정 완료!
echo ============================================================
echo.
echo Python 버전: %PYTHON_VER%
echo.
echo 다음 명령어로 가상환경을 활성화하세요:
echo   venv\Scripts\activate
echo.
echo 또는 이 스크립트를 실행하면 자동으로 활성화됩니다.
echo.
pause
