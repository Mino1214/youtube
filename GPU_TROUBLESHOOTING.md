# GPU 호환성 문제 해결 가이드

## 문제 상황
`RuntimeError: CUDA error: no kernel image is available for execution on the device`

이 오류는 PyTorch가 현재 GPU의 compute capability를 지원하지 않아서 발생합니다.

## ✅ 즉시 해결 방법 (CPU 모드 사용)

프로그램이 이제 **자동으로 CPU 모드로 전환**됩니다!

`config.yaml` 파일에서 다음 설정이 활성화되어 있는지 확인하세요:

```yaml
video_generation:
  use_gpu: true
  force_gpu: false  # ✅ false로 설정
  auto_cpu_fallback: true  # ✅ true로 설정 (자동 CPU fallback)
```

**설정 완료 후 프로그램을 다시 실행하면 GPU 오류 시 자동으로 CPU 모드로 전환됩니다.**

⚠️ **참고**: CPU 모드는 GPU보다 처리 속도가 느립니다. 특히 비디오 생성 시 시간이 오래 걸릴 수 있습니다.

---

## 🔧 근본적인 해결 방법

### 1. PyTorch 재설치

현재 설치된 PyTorch 버전이 GPU를 지원하지 않을 수 있습니다. 최신 버전으로 재설치하세요:

#### 방법 A: 안정 버전 (권장)

```powershell
# 기존 PyTorch 제거
pip uninstall torch torchvision torchaudio -y

# 최신 안정 버전 설치 (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 방법 B: Nightly 버전 (더 많은 GPU 지원)

```powershell
# 기존 PyTorch 제거
pip uninstall torch torchvision torchaudio -y

# Nightly 버전 설치 (최신 CUDA 지원)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

### 2. GPU 정보 확인

PyTorch 재설치 후 GPU가 제대로 인식되는지 확인하세요:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('Compute Capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A')"
```

### 3. NVIDIA 드라이버 업데이트

GPU 드라이버가 오래된 경우 업데이트하세요:

1. `nvidia-smi` 명령으로 현재 드라이버 버전 확인
2. [NVIDIA 공식 사이트](https://www.nvidia.com/Download/index.aspx)에서 최신 드라이버 다운로드
3. 드라이버 설치 후 재부팅

---

## 📊 GPU Compute Capability 요구사항

- **최소 요구사항**: Compute Capability 7.0 이상 (Volta 아키텍처 이상)
- **권장 사양**: Compute Capability 8.0 이상 (Ampere, Ada Lovelace 아키텍처)

### GPU 세대별 Compute Capability

| GPU 아키텍처 | Compute Capability | 예시 GPU |
|------------|-------------------|---------|
| Turing | 7.5 | RTX 2060, 2070, 2080 |
| Ampere | 8.0, 8.6 | RTX 3060, 3070, 3080, 3090 |
| Ada Lovelace | 8.9 | RTX 4060, 4070, 4080, 4090 |
| Hopper | 9.0 | H100 |

---

## 🔍 추가 문제 해결

### 문제: PyTorch 재설치 후에도 오류 발생

**해결책**: CUDA Toolkit 버전 확인

```powershell
# CUDA 버전 확인
nvcc --version

# PyTorch CUDA 버전 확인
python -c "import torch; print('PyTorch CUDA version:', torch.version.cuda)"
```

두 버전이 호환되어야 합니다. 호환되지 않으면 적절한 PyTorch 버전을 설치하세요:

- CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- CUDA 12.4: `--index-url https://download.pytorch.org/whl/cu124`

### 문제: 구형 GPU 사용

Compute Capability 7.0 미만의 GPU는 최신 PyTorch에서 지원하지 않을 수 있습니다.

**해결책**: 
1. CPU 모드 사용 (`auto_cpu_fallback: true`)
2. 또는 PyTorch 1.x 버전 사용 (권장하지 않음)

---

## ✅ 설정 확인 체크리스트

- [ ] `config.yaml`에서 `auto_cpu_fallback: true` 설정
- [ ] `config.yaml`에서 `force_gpu: false` 설정
- [ ] NVIDIA 드라이버가 최신 버전인지 확인 (`nvidia-smi`)
- [ ] PyTorch가 CUDA를 인식하는지 확인
- [ ] GPU Compute Capability가 7.0 이상인지 확인

---

## 📞 추가 도움말

더 자세한 정보는 다음 링크를 참조하세요:

- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-downloads)
- [NVIDIA 드라이버 다운로드](https://www.nvidia.com/Download/index.aspx)
