# 빠른 시작 가이드

## 🚀 전체 프로세스 (처음부터)

### 1단계: 가상환경 완전 재생성 (권장)
```bash
clean_venv.bat
```
이 스크립트가:
- 기존 가상환경 삭제
- 새 가상환경 생성
- 호환 버전으로 모든 패키지 설치

**시간**: 약 10-30분 (인터넷 속도에 따라)

---

### 2단계: 모델 다운로드
```bash
venv\Scripts\activate
python download_all_models.py --auto
```
`--auto` 옵션으로 자동 다운로드 (밤새 돌려놓을 수 있음)

**시간**: 모델 크기에 따라 수 시간 (밤새 권장)

---

### 3단계: 비디오 생성
```bash
venv\Scripts\activate
python main.py
```

---

## 🔧 문제 해결

### 버전 충돌이 발생한 경우
```bash
fix_version_conflicts.bat
```

### 의존성만 재설치
```bash
install_dependencies.bat
```

---

## 📝 상세 실행 방법

### 방법 1: 완전 초기화 (가장 확실)
```bash
# 1. 가상환경 재생성
clean_venv.bat

# 2. 가상환경 활성화
venv\Scripts\activate

# 3. 모델 다운로드
python download_all_models.py --auto

# 4. 비디오 생성
python main.py
```

### 방법 2: 기존 환경에서 충돌만 해결
```bash
# 1. 버전 충돌 해결
fix_version_conflicts.bat

# 2. 가상환경 활성화
venv\Scripts\activate

# 3. 모델 다운로드 (필요시)
python download_all_models.py --auto

# 4. 비디오 생성
python main.py
```

---

## ⚠️ 주의사항

1. **가상환경 활성화 필수**
   - 모든 Python 명령어 실행 전에 `venv\Scripts\activate` 실행
   - 활성화되면 프롬프트에 `(venv)` 표시됨

2. **모델 다운로드**
   - 첫 실행 시 모든 모델을 다운로드해야 함
   - `--auto` 옵션으로 자동 진행 (밤새 돌려놓기 가능)
   - 디스크 공간: 약 50GB 필요

3. **입력 파일**
   - `input.txt` 또는 `input_text.txt` 파일에 영어 텍스트 작성
   - 이 텍스트가 비디오로 변환됨

---

## 🎯 빠른 체크리스트

- [ ] `clean_venv.bat` 실행 완료
- [ ] 가상환경 활성화 (`venv\Scripts\activate`)
- [ ] `python download_all_models.py --auto` 실행 완료
- [ ] `input.txt` 파일에 텍스트 작성
- [ ] `python main.py` 실행

---

## 💡 팁

- **밤새 돌리기**: 모델 다운로드는 `--auto` 옵션으로 자동 진행
- **에러 발생 시**: `fix_version_conflicts.bat` 실행 후 재시도
- **디스크 공간**: 모델 저장 공간 확인 (약 50GB)
