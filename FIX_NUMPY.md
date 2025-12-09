# NumPy 오류 해결 가이드

## 문제
`RuntimeError: Numpy is not available` 오류가 발생합니다.

## 원인
- NumPy DLL 로드 실패
- PyTorch와 NumPy 버전 호환성 문제
- 여러 NumPy 버전이 설치되어 충돌

## 해결 방법

### 방법 1: 자동 수정 스크립트 (권장)
```bash
python fix_numpy.py
```

### 방법 2: 수동 해결

#### 1단계: NumPy 제거 및 재설치
```bash
pip uninstall numpy -y
pip install numpy>=1.24.0
```

#### 2단계: PyTorch 재설치 (필요시)
```bash
pip uninstall torch -y
pip install torch
```

#### 3단계: 호환 버전으로 고정
```bash
pip install numpy==1.24.0 torch
```

### 방법 3: 전체 패키지 재설치
```bash
pip uninstall numpy torch -y
pip install -r requirements.txt
```

## 확인 방법

다음 명령어로 테스트:
```python
python -c "import numpy; import torch; arr = numpy.array([1,2,3]); tensor = torch.from_numpy(arr); print('Success!')"
```

성공하면 "Success!"가 출력됩니다.

## Windows 특별 주의사항

Windows에서 NumPy DLL 문제가 자주 발생합니다:
1. Visual C++ Redistributable 설치 확인
2. Python 버전과 NumPy 버전 호환성 확인
3. 가상환경 사용 권장

## 추가 정보

- NumPy 권장 버전: 1.24.0 ~ 1.26.x
- PyTorch와 호환되는 NumPy 버전 사용
- CPU 전용 PyTorch의 경우 NumPy 1.24.0 권장
