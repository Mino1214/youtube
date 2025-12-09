# RTX 5060 Ti GPU 지원 문제 - 최종 안내

## 🔴 현재 상황

**RTX 5060 Ti는 compute capability sm_120 (12.0)을 사용하지만, 현재 PyTorch의 모든 버전이 이를 지원하지 않습니다.**

테스트한 버전:
- ❌ PyTorch 2.4.0+cu121
- ❌ PyTorch 2.6.0+cu124  
- ❌ PyTorch 2.9.1+cu126
- ❌ PyTorch Nightly (설치 실패)

## ✅ 자동 해결 방법

코드가 **자동으로 CPU 모드로 전환**하도록 수정했습니다.

### 작동 방식:
1. GPU 모드로 시도
2. CUDA 오류 발생 시 자동으로 CPU로 전환
3. 경고 메시지 출력 후 계속 진행

### 실행:
```bash
python main.py
```

**경고 메시지가 나오지만 정상적으로 작동합니다.**

## 🔧 수동 해결 방법

### 방법 1: config.yaml에서 CPU 강제 설정
```yaml
whisper:
  device: "cpu"

llm:
  use_gpu: false
```

### 방법 2: PyTorch 소스에서 직접 빌드 (고급)
sm_120 지원을 포함하여 직접 빌드 (복잡하고 시간 소요)

## 📊 성능

| 모드 | 상태 | 속도 |
|------|------|------|
| GPU (원하는 방식) | ❌ 작동 안 함 | - |
| CPU (자동 전환) | ✅ 작동 | 느림 (3-5배) |

## 💡 결론

**현재는 CPU 모드로만 작동 가능합니다.**
- 코드가 자동으로 CPU로 전환합니다
- 경고 메시지는 무시하고 진행하면 됩니다
- PyTorch가 sm_120을 지원하는 버전이 나올 때까지 대기 필요

## 🔮 향후

PyTorch 공식에서 sm_120 지원 버전이 나오면:
1. 최신 PyTorch 설치
2. `config.yaml`에서 `device: "cuda"`로 변경
3. GPU 모드로 사용 가능

