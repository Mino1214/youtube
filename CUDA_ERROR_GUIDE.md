# CUDA 오류 해결 가이드

## 🔴 "no kernel image is available for execution on the device" 오류

이 오류는 PyTorch가 현재 GPU의 **compute capability**를 지원하지 않을 때 발생합니다.

## 🔍 원인 진단

### 1. GPU 호환성 확인
```bash
check_gpu_compatibility.bat
```

이 스크립트가 다음을 확인합니다:
- GPU 이름
- Compute Capability
- CUDA 버전
- PyTorch 버전
- GPU 연산 테스트

### 2. GPU Compute Capability 확인
```bash
venv\Scripts\activate
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0)); print('Capability:', torch.cuda.get_device_capability(0))"
```

## ✅ 해결 방법

### 방법 1: PyTorch Nightly 버전 설치 (권장)
```bash
install_pytorch_nightly.bat
```

Nightly 버전은 최신 GPU compute capability를 지원합니다.

### 방법 2: PyTorch 소스에서 빌드
1. PyTorch 소스 다운로드
2. GPU compute capability에 맞게 빌드
3. 설치

### 방법 3: 다른 CUDA 버전 시도
```bash
venv\Scripts\activate
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 Compute Capability 호환성

| Compute Capability | GPU 세대 | PyTorch 지원 |
|-------------------|---------|-------------|
| 8.0+ | Ampere, Ada, Hopper | ✅ 완전 지원 |
| 7.5 | Turing | ✅ 완전 지원 |
| 7.0 | Volta | ✅ 대부분 지원 |
| 6.0-6.2 | Pascal | ⚠️ 제한적 지원 |
| 5.0-5.2 | Maxwell | ❌ 지원 안 함 |

## 🔧 순차적 오류 제어

코드는 다음 순서로 오류를 처리합니다:

1. **초기화 단계**: GPU 초기화 테스트
2. **모델 로드 단계**: 모델을 GPU로 이동 시도
3. **실행 단계**: 실제 연산 시 오류 감지

각 단계에서 오류가 발생하면:
- 명확한 오류 메시지 표시
- 해결 방법 제시
- **CPU로 전환하지 않음** (GPU만 사용)

## 💡 디버깅 팁

### CUDA_LAUNCH_BLOCKING 환경 변수
```bash
set CUDA_LAUNCH_BLOCKING=1
python main.py
```

이렇게 하면 CUDA 오류가 즉시 보고됩니다.

### GPU 정보 상세 확인
```bash
nvidia-smi
```

GPU 드라이버 버전과 CUDA 버전을 확인합니다.

## 🚫 CPU 사용 안 함

이 프로젝트는 **GPU만 사용**합니다. CPU fallback이 없으므로:
- GPU 오류 발생 시 명확한 오류 메시지 표시
- 해결 방법 제시
- 프로그램 종료 (CPU로 자동 전환하지 않음)
