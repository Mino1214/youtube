# RTX 5060 Ti 호환성 문제 해결 가이드

## 🔴 문제

RTX 5060 Ti는 compute capability sm_120 (12.0)을 사용하지만, 현재 PyTorch 2.4.0+cu121은 sm_50~sm_90까지만 지원합니다.

**오류 메시지:**
```
CUDA error: no kernel image is available for execution on the device
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible
```

## ✅ 해결 방법

### 방법 1: PyTorch 업그레이드 (권장) ⭐

**Windows:**
```powershell
# 업그레이드 스크립트 실행
.\upgrade_pytorch.bat

# 또는 수동으로
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Ubuntu:**
```bash
bash upgrade_pytorch.sh

# 또는 수동으로
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 방법 2: CPU 모드로 실행 (임시 해결책)

`config.yaml` 파일 수정:
```yaml
whisper:
  device: "cpu"  # "cuda" 대신 "cpu"

llm:
  use_gpu: false  # GPU 사용 안 함
```

**주의:** CPU 모드는 훨씬 느립니다!

### 방법 3: Nightly 빌드 사용 (최신 기능)

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

## 🔍 확인

업그레이드 후 확인:
```bash
python check_cuda.py
```

**성공 시:**
- 경고 메시지가 사라짐
- "✅ CUDA 작동 정상!" 메시지 출력

## 📝 참고

- PyTorch 2.5.0 이상이 sm_120을 지원합니다
- CUDA 12.4 또는 12.6 버전 사용 권장
- 자동 CPU fallback 기능이 활성화되어 있어 일부 모듈은 자동으로 CPU로 전환됩니다

## 🚀 빠른 해결

가장 빠른 해결책:
```bash
# Windows
.\upgrade_pytorch.bat

# 확인
python check_cuda.py
```

