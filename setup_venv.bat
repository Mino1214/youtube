@echo off
REM Python 3.10 가상환경 설정 스크립트 (Windows)

echo ============================================================
echo Python 3.10 가상환경 설정
echo ============================================================
echo.

REM Python 3.10 확인
echo [1/5] Python 3.10 확인 중...
py -3.10 --version
if errorlevel 1 (
    echo.
    echo ❌ Python 3.10을 찾을 수 없습니다.
    echo Python 3.10을 설치하거나 PATH에 추가해주세요.
    pause
    exit /b 1
)
echo ✅ Python 3.10 확인 완료
echo.

REM 기존 가상환경 삭제 (선택사항)
if exist venv (
    echo [2/5] 기존 가상환경 삭제 중...
    rmdir /s /q venv
    echo ✅ 기존 가상환경 삭제 완료
    echo.
) else (
    echo [2/5] 기존 가상환경 없음
    echo.
)

REM 가상환경 생성
echo [3/5] Python 3.10 가상환경 생성 중...
py -3.10 -m venv venv
if errorlevel 1 (
    echo.
    echo ❌ 가상환경 생성 실패
    pause
    exit /b 1
)
echo ✅ 가상환경 생성 완료
echo.

REM 가상환경 활성화 및 pip 업그레이드
echo [4/5] pip 업그레이드 중...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo ✅ pip 업그레이드 완료
echo.

REM 필수 패키지 설치
echo [5/5] 필수 패키지 설치 중...
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
echo 다음 명령어로 가상환경을 활성화하세요:
echo   venv\Scripts\activate
echo.
echo 또는 이 스크립트를 실행하면 자동으로 활성화됩니다.
echo.
pause
