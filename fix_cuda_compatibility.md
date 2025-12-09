# RTX 5060 Ti CUDA 호환성 문제 해결

## 문제
RTX 5060 Ti는 compute capability sm_120 (12.0)을 사용하지만, 현재 PyTorch 2.4.0+cu121은 sm_50~sm_90까지만 지원합니다.

## 해결 방법

### 방법 1: PyTorch 최신 버전으로 업그레이드 (권장)

```bash
# 기존 PyTorch 제거
pip uninstall torch torchvision torchaudio

# 최신 PyTorch 설치 (CUDA 12.4 또는 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

또는 Nightly 빌드 (최신 기능 포함):
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

### 방법 2: CPU 모드로 실행

`config.yaml` 파일 수정:
```yaml
whisper:
  device: "cpu"  # "cuda" 대신 "cpu"

llm:
  use_gpu: false  # GPU 사용 안 함
```

### 방법 3: CUDA 12.6 버전 사용

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 확인

업그레이드 후 확인:
```bash
python check_cuda.py
```

경고 메시지가 사라지고 "✅ CUDA 작동 정상!"이 나오면 성공입니다.

