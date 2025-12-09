# espeak-ng 환경 변수 설정 가이드 (Chocolatey 설치용)

## 🎯 빠른 해결 방법

Chocolatey로 설치했다면, 다음 3단계만 하면 됩니다!

---

## ✅ 1단계: espeak-ng 설치 확인

PowerShell에서:

```powershell
# espeak-ng가 설치되어 있는지 확인
choco list --local-only | findstr espeak
```

**출력 예시:**
```
espeak-ng 1.51.0
```

---

## ✅ 2단계: 설치 경로 확인

Chocolatey로 설치하면 보통 다음 경로에 설치됩니다:

```powershell
# 설치 경로 확인
dir "C:\Program Files\eSpeak NG"
```

**또는:**

```powershell
dir "C:\ProgramData\chocolatey\lib\espeak-ng\tools"
```

---

## ✅ 3단계: 환경 변수 PATH에 추가

### 방법 A: PowerShell로 자동 추가 (가장 빠름!) ⭐

**관리자 권한으로 PowerShell 실행 후:**

```powershell
# espeak-ng 경로를 시스템 PATH에 추가
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "Machine") + ";C:\Program Files\eSpeak NG",
    "Machine"
)

# 현재 세션에도 즉시 적용
$env:Path += ";C:\Program Files\eSpeak NG"

echo "✅ 환경 변수 추가 완료!"
```

**또는 Chocolatey 경로로 설치된 경우:**

```powershell
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "Machine") + ";C:\ProgramData\chocolatey\lib\espeak-ng\tools",
    "Machine"
)

$env:Path += ";C:\ProgramData\chocolatey\lib\espeak-ng\tools"

echo "✅ 환경 변수 추가 완료!"
```

---

### 방법 B: GUI로 수동 추가

1. **`Windows 키`** 누르고 **"환경 변수"** 검색

2. **"시스템 환경 변수 편집"** 클릭

3. **"환경 변수"** 버튼 클릭

4. **"시스템 변수"** 섹션에서 **"Path"** 선택

5. **"편집"** 클릭

6. **"새로 만들기"** 클릭

7. 다음 경로 중 하나 입력:
   ```
   C:\Program Files\eSpeak NG
   ```
   
   **또는:**
   ```
   C:\ProgramData\chocolatey\lib\espeak-ng\tools
   ```

8. **"확인"** 클릭 (3번)

---

## ✅ 4단계: 설치 확인

**새 PowerShell 창**을 열고:

```powershell
# espeak-ng 버전 확인
espeak-ng --version
```

**성공 출력:**
```
eSpeak NG text-to-speech: 1.51
```

**실패하면:**
```
'espeak-ng'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는 배치 파일이 아닙니다.
```
→ PowerShell을 **완전히 닫고** 새로 열어보세요!

---

## 🔍 문제 해결

### Q: "espeak-ng --version"이 작동하지 않아요

**해결책 1: PowerShell 완전히 재시작**
- 모든 PowerShell 창 닫기
- 새 PowerShell 창 열기
- `espeak-ng --version` 다시 시도

**해결책 2: 컴퓨터 재부팅**
- 환경 변수 변경 후 재부팅하면 100% 적용됨

**해결책 3: 경로 직접 확인**
```powershell
# 실제 설치 경로 찾기
Get-ChildItem "C:\Program Files" -Recurse -Filter "espeak-ng.exe" -ErrorAction SilentlyContinue | Select-Object FullName

Get-ChildItem "C:\ProgramData\chocolatey" -Recurse -Filter "espeak-ng.exe" -ErrorAction SilentlyContinue | Select-Object FullName
```

찾은 경로의 **상위 폴더**를 PATH에 추가하세요.

---

### Q: 관리자 권한이 없어요

**해결책: 사용자 환경 변수에 추가**

```powershell
# 현재 사용자만 적용 (관리자 권한 불필요)
[Environment]::SetEnvironmentVariable(
    "Path",
    [Environment]::GetEnvironmentVariable("Path", "User") + ";C:\Program Files\eSpeak NG",
    "User"
)

$env:Path += ";C:\Program Files\eSpeak NG"
```

---

### Q: 경로를 찾을 수 없어요

**해결책: Chocolatey 재설치**

```powershell
# 관리자 권한 PowerShell에서
choco uninstall espeak-ng -y
choco install espeak-ng -y

# 설치 후 경로 확인
where.exe espeak-ng
```

---

## 🎯 한 번에 해결하는 스크립트

**관리자 권한 PowerShell**에서 복사-붙여넣기:

```powershell
# espeak-ng 경로 찾기
$paths = @(
    "C:\Program Files\eSpeak NG",
    "C:\ProgramData\chocolatey\lib\espeak-ng\tools"
)

$found = $false
foreach ($path in $paths) {
    if (Test-Path "$path\espeak-ng.exe") {
        Write-Host "✅ espeak-ng 찾음: $path" -ForegroundColor Green
        
        # PATH에 추가
        $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
        if ($currentPath -notlike "*$path*") {
            [Environment]::SetEnvironmentVariable(
                "Path",
                "$currentPath;$path",
                "Machine"
            )
            Write-Host "✅ 환경 변수에 추가 완료!" -ForegroundColor Green
        } else {
            Write-Host "✅ 이미 PATH에 있습니다!" -ForegroundColor Yellow
        }
        
        # 현재 세션에 적용
        $env:Path += ";$path"
        
        $found = $true
        break
    }
}

if (-not $found) {
    Write-Host "❌ espeak-ng를 찾을 수 없습니다. 다시 설치하세요:" -ForegroundColor Red
    Write-Host "choco install espeak-ng -y" -ForegroundColor Yellow
}

# 테스트
Write-Host "`n테스트 중..." -ForegroundColor Cyan
espeak-ng --version
```

---

## ✅ 체크리스트

- [ ] Chocolatey로 espeak-ng 설치 확인
- [ ] 설치 경로 확인 (Program Files 또는 Chocolatey)
- [ ] PowerShell 관리자 권한으로 실행
- [ ] 환경 변수 PATH에 추가 명령 실행
- [ ] PowerShell 완전히 재시작
- [ ] `espeak-ng --version` 작동 확인
- [ ] Coqui TTS 테스트

---

## 🎉 완료 후 테스트

환경 변수 설정이 완료되면:

```powershell
# 1. espeak-ng 테스트
espeak-ng --version

# 2. Coqui TTS 테스트
python -c "from TTS.api import TTS; print('✅ Coqui TTS 사용 가능!')"

# 3. 음성 생성 테스트
python -c "from TTS.api import TTS; tts = TTS(model_name='tts_models/en/vctk/vits', progress_bar=False, gpu=False); tts.tts_to_file(text='Hello from Coqui TTS', file_path='test_coqui.wav', speaker='p245'); print('✅ 테스트 음성 생성 완료!')"

# 4. 프로그램 실행
python main.py
```

---

## 💡 팁

- **PowerShell 재시작 꼭 하세요!** 환경 변수는 새 세션에서만 적용됩니다.
- 재시작 후에도 안 되면 **컴퓨터를 재부팅**하세요.
- 여전히 안 되면 **edge-tts를 대신 사용**하세요 (espeak-ng 불필요):
  ```powershell
  pip install edge-tts
  python main.py
  ```

---

## 🔗 관련 링크

- [espeak-ng GitHub](https://github.com/espeak-ng/espeak-ng)
- [Chocolatey espeak-ng 패키지](https://community.chocolatey.org/packages/espeak-ng)
