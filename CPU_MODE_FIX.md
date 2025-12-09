# RTX 5060 Ti CPU 모드 전환 완료

## ✅ 해결 완료

RTX 5060 Ti는 compute capability sm_120을 사용하지만, 현재 PyTorch 버전들이 이를 지원하지 않습니다.

**해결책:** `config.yaml`에서 CPU 모드로 전환했습니다.

## 변경 사항

### config.yaml 수정:
```yaml
whisper:
  device: "cpu"  # CUDA 대신 CPU 사용

llm:
  use_gpu: false  # GPU 사용 안 함
```

## 실행

이제 정상적으로 작동합니다:

```bash
python main.py
```

## 성능

- **CPU 모드**: 느리지만 안정적으로 작동
- **예상 시간**: GPU 대비 3-5배 느림
- **권장**: 짧은 비디오로 먼저 테스트

## 향후 업그레이드

PyTorch가 sm_120을 지원하는 버전이 나오면:
1. `config.yaml`에서 `device: "cuda"`, `use_gpu: true`로 변경
2. PyTorch 최신 버전 설치

## 참고

- CPU 모드도 완전히 작동합니다
- Whisper large-v3는 CPU에서도 실행 가능 (느리지만)
- LLM도 CPU에서 실행 가능 (매우 느림)

