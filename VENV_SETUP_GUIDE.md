# Python 3.10 가상환경 설정 가이드

## 🚀 빠른 시작

### Windows:
```bash
setup_venv.bat
```

### Linux/Mac:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

## 📝 수동 설정 방법

### 1. 가상환경 생성
```bash
py -3.10 -m venv venv
```

### 2. 가상환경 활성화

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. 필수 패키지 설치
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# NumPy (호환 버전)
pip install numpy==1.24.0

# PyTorch
pip install torch

# 모든 의존성
pip install -r requirements.txt
```

## ✅ 확인

가상환경이 활성화되면 프롬프트 앞에 `(venv)`가 표시됩니다.

```bash
(venv) C:\Users\alsdh\OneDrive\Desktop\aivideo>
```

## 🔧 사용 방법

### 가상환경 활성화 후:
```bash
# 1. 모델 다운로드
python download_all_models.py --auto

# 2. 영어 TTS 모델 다운로드
python download_english_tts.py

# 3. 프로그램 실행
python main.py
```

### 가상환경 비활성화:
```bash
deactivate
```

## 💡 장점

1. **깨끗한 환경**: 기존 패키지 충돌 없음
2. **Python 3.10**: 안정적인 버전
3. **독립적**: 시스템 Python과 분리
4. **재현 가능**: 동일한 환경 보장

## ⚠️ 주의사항

- 가상환경을 활성화한 상태에서만 패키지가 설치됩니다
- 매번 작업 전에 `venv\Scripts\activate` 실행 필요
- 가상환경은 프로젝트 폴더 내 `venv/` 디렉토리에 생성됩니다

## 🐛 문제 해결

### Python 3.10을 찾을 수 없음
```bash
# Python 3.10 설치 확인
py -3.10 --version

# 또는 python3.10
python3.10 --version
```

### 가상환경 활성화 실패
- Windows: `venv\Scripts\activate.bat` 직접 실행
- Linux/Mac: `source venv/bin/activate` 사용

### 패키지 설치 오류
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 개별 설치
pip install numpy==1.24.0
pip install torch
```
