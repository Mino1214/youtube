# espeak-ng 설치 가이드 (Coqui TTS용)

## 문제 상황

```
[!] No espeak backend found. Install espeak-ng or espeak to your system.
```

Coqui TTS를 사용하려면 **espeak-ng**가 필요합니다!

---

## ✅ Windows 설치 방법 (빠른 해결)

### 방법 1: Chocolatey로 설치 (가장 간단!) ⭐

```powershell
# Chocolatey가 설치되어 있다면
choco install espeak-ng

# 설치 후 PowerShell 재시작
```

### 방법 2: 직접 다운로드 (권장)

1. **다운로드**
   - [espeak-ng GitHub 릴리스 페이지](https://github.com/espeak-ng/espeak-ng/releases) 방문
   - 최신 버전의 `espeak-ng-X64.msi` 다운로드 (예: `espeak-ng-1.51-x64.msi`)

2. **설치**
   - 다운로드한 `.msi` 파일 실행
   - "Next" 클릭하여 설치 진행
   - 기본 경로로 설치: `C:\Program Files\eSpeak NG\`

3. **환경 변수 설정** (중요!)
   
   **자동 설정 (PowerShell 관리자 권한으로 실행):**
   ```powershell
   # PowerShell을 관리자 권한으로 실행한 후
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\eSpeak NG", "Machine")
   
   # 또는 PATH에 추가
   $env:PATH += ";C:\Program Files\eSpeak NG"
   ```

   **수동 설정:**
   - `Windows 키` + `검색` → "환경 변수" 입력
   - "시스템 환경 변수 편집" 클릭
   - "환경 변수" 버튼 클릭
   - "시스템 변수"에서 "Path" 선택 → "편집" 클릭
   - "새로 만들기" → `C:\Program Files\eSpeak NG` 입력
   - "확인" 클릭

4. **PowerShell 재시작**
   - 현재 PowerShell 종료
   - 새 PowerShell 창 열기

5. **설치 확인**
   ```powershell
   espeak-ng --version
   ```
   
   출력 예시:
   ```
   eSpeak NG text-to-speech: 1.51
   ```

---

## 🚀 빠른 설치 스크립트 (권장!)

PowerShell을 **관리자 권한**으로 실행한 후:

```powershell
# 1. Chocolatey 설치 (없다면)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# 2. espeak-ng 설치
choco install espeak-ng -y

# 3. 설치 확인
espeak-ng --version
```

---

## 🎯 설치 후 확인

### 1. espeak-ng 테스트
```powershell
espeak-ng --version
```

### 2. Coqui TTS 테스트
```powershell
python -c "from TTS.api import TTS; tts = TTS(model_name='tts_models/en/vctk/vits', progress_bar=False, gpu=False); print('✅ Coqui TTS 설치 성공!')"
```

### 3. 음성 생성 테스트
```powershell
python -c "from TTS.api import TTS; tts = TTS(model_name='tts_models/en/vctk/vits', progress_bar=False, gpu=False); tts.tts_to_file(text='Hello, this is a test of Coqui TTS with espeak-ng', file_path='test_coqui.wav', speaker='p245'); print('✅ 음성 파일 생성 완료: test_coqui.wav')"
```

---

## 📁 설치 경로

espeak-ng가 설치되는 경로:
- **기본 경로**: `C:\Program Files\eSpeak NG\`
- **실행 파일**: `C:\Program Files\eSpeak NG\espeak-ng.exe`
- **데이터 파일**: `C:\Program Files\eSpeak NG\espeak-ng-data\`

---

## ❌ 문제 해결

### Q: "espeak-ng를 찾을 수 없습니다" 오류
A: 환경 변수 PATH에 제대로 추가되지 않았습니다.

**해결:**
```powershell
# PowerShell에서 현재 세션에만 적용 (임시)
$env:PATH += ";C:\Program Files\eSpeak NG"

# 영구 적용 (관리자 권한 필요)
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\eSpeak NG", "Machine")
```

### Q: "espeak-ng --version"이 작동하지 않음
A: PowerShell을 재시작하세요.

### Q: 관리자 권한이 없어요
A: 
1. 수동으로 `.msi` 파일 다운로드 및 설치 (관리자 권한 필요)
2. 또는 **edge-tts**를 대신 사용 (설치 불필요)
   ```powershell
   pip install edge-tts
   python main.py
   ```

### Q: Chocolatey 설치가 실패해요
A: 수동 다운로드 방법 사용:
1. https://github.com/espeak-ng/espeak-ng/releases
2. `espeak-ng-X64.msi` 다운로드
3. 실행하여 설치
4. 환경 변수 PATH에 추가

---

## 🎉 설치 완료 후

espeak-ng 설치가 완료되면:

```powershell
# Coqui TTS 설치 (아직 안 했다면)
pip install TTS

# 프로그램 실행
python main.py
```

이제 **Coqui TTS**가 정상 작동하여 **최고 품질의 자연스러운 음성**이 생성됩니다! 🎤

---

## 💡 대안 (espeak-ng 설치가 어려운 경우)

espeak-ng 설치가 어렵다면 다른 TTS 엔진 사용:

### 옵션 1: edge-tts (권장 대안) 🥈
```powershell
pip install edge-tts
python main.py
```
- 설치 간단 (espeak-ng 불필요)
- 매우 자연스러운 음성
- 인터넷 연결 필요

### 옵션 2: gTTS ⚡
```powershell
pip install gtts
python main.py
```
- 가장 간단한 설치
- 빠른 처리
- 인터넷 연결 필요

---

## ✅ 체크리스트

- [ ] espeak-ng 다운로드 완료
- [ ] espeak-ng 설치 완료
- [ ] 환경 변수 PATH에 추가 완료
- [ ] PowerShell 재시작
- [ ] `espeak-ng --version` 확인
- [ ] Coqui TTS 테스트 성공
- [ ] 프로그램 실행: `python main.py`

---

## 🔗 관련 링크

- [espeak-ng GitHub](https://github.com/espeak-ng/espeak-ng)
- [espeak-ng 릴리스 페이지](https://github.com/espeak-ng/espeak-ng/releases)
- [Coqui TTS 문서](https://github.com/coqui-ai/TTS)

---

## 🎯 요약

**가장 빠른 방법:**
1. [espeak-ng 다운로드](https://github.com/espeak-ng/espeak-ng/releases) → `espeak-ng-X64.msi`
2. 설치 실행 (기본 경로)
3. 환경 변수 PATH에 `C:\Program Files\eSpeak NG` 추가
4. PowerShell 재시작
5. `python main.py` 실행 → ✅ 완료!

**어려우면 대신 이거:**
```powershell
pip install edge-tts
python main.py
```
