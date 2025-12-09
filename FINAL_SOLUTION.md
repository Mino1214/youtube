# RTX 5060 Ti 최종 해결 방안

## 🔴 문제 상황

RTX 5060 Ti는 **compute capability sm_120 (12.0)**을 사용하지만, 현재 PyTorch의 모든 안정 버전이 이를 지원하지 않습니다.

## ✅ 해결 방안

### 옵션 1: CPU 모드 사용 (현재 설정) ⭐ **권장**

**장점:**
- 즉시 작동
- 안정적
- 추가 설정 불필요

**단점:**
- GPU보다 느림 (3-5배)

**현재 상태:** 이미 `config.yaml`에서 CPU 모드로 설정됨

### 옵션 2: PyTorch Nightly 빌드 시도

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

**주의:** Nightly 빌드는 불안정할 수 있음

### 옵션 3: PyTorch 소스에서 빌드 (고급)

sm_120 지원을 포함하여 직접 빌드 (복잡하고 시간 소요)

### 옵션 4: 대기

PyTorch 공식에서 sm_120 지원 버전 출시 대기

## 🎯 권장 사항

**현재는 CPU 모드로 사용하는 것을 권장합니다:**

1. ✅ 이미 설정 완료 (`config.yaml`)
2. ✅ 안정적으로 작동
3. ✅ 추가 설치 불필요

```bash
# 바로 실행 가능
python main.py
```

## 📊 성능 비교

| 모드 | 속도 | 안정성 | 설정 |
|------|------|--------|------|
| CPU | 느림 (3-5배) | ⭐⭐⭐⭐⭐ | 완료 |
| GPU (미지원) | 빠름 | ❌ 작동 안 함 | 불가 |
| Nightly | 빠름 | ⭐⭐ | 시도 필요 |

## 💡 결론

**지금 당장 사용하려면:** CPU 모드 사용 (이미 설정됨)
**향후 GPU 사용:** PyTorch 공식 지원 대기 또는 Nightly 빌드 시도

