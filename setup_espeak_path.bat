@echo off
chcp 65001 > nul
echo.
echo ============================================================
echo espeak-ng 환경 변수 PATH 추가
echo ============================================================
echo.

REM 관리자 권한 확인
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ 관리자 권한이 필요합니다!
    echo.
    echo 해결 방법:
    echo 1. 이 파일을 우클릭
    echo 2. "관리자 권한으로 실행" 선택
    echo.
    pause
    exit /b 1
)

echo ✅ 관리자 권한 확인 완료
echo.

REM espeak-ng 설치 확인
set "ESPEAK_PATH=C:\Program Files\eSpeak NG"
if not exist "%ESPEAK_PATH%\espeak-ng.exe" (
    echo ❌ espeak-ng를 찾을 수 없습니다: %ESPEAK_PATH%
    echo.
    echo 먼저 설치하세요:
    echo choco install espeak-ng -y
    echo.
    pause
    exit /b 1
)

echo ✅ espeak-ng 찾음: %ESPEAK_PATH%
echo.

REM 환경 변수 PATH에 추가
echo 환경 변수 PATH에 추가 중...
setx /M PATH "%PATH%;%ESPEAK_PATH%" > nul 2>&1

if %errorLevel% equ 0 (
    echo ✅ 환경 변수 PATH에 추가 완료!
    echo.
    echo ============================================================
    echo 완료!
    echo ============================================================
    echo.
    echo 다음 단계:
    echo 1. 이 창을 닫기
    echo 2. 모든 PowerShell 창 닫기
    echo 3. 새 PowerShell 열기
    echo 4. 테스트: espeak-ng --version
    echo 5. 프로그램 실행: python main.py
    echo.
) else (
    echo ❌ 환경 변수 추가 실패
    echo.
)

pause
