# 의존성 문제 해결 가이드

## 🔴 현재 문제

다음과 같은 에러가 발생하는 경우:

```
cannot import name 'CLIPImageProcessor' from 'transformers'
```

또는

```
Failed to import diffusers.pipelines...
```

## ✅ 해결 방법

### 방법 1: 자동 수정 스크립트 실행 (추천)

```bash
python fix_dependencies.py
```

이 스크립트가 필요한 패키지들을 자동으로 최신 버전으로 업그레이드합니다.

### 방법 2: 수동 업그레이드

```bash
pip install --upgrade transformers>=4.40.0
pip install --upgrade diffusers>=0.27.0
pip install --upgrade huggingface-hub>=0.20.0
pip install --upgrade accelerate>=0.24.0
pip install --upgrade safetensors>=0.4.0
pip install --upgrade invisible-watermark>=0.2.0
```

### 방법 3: 전체 재설치

```bash
# 기존 패키지 제거
pip uninstall transformers diffusers huggingface-hub accelerate -y

# 최신 버전 설치
pip install transformers>=4.40.0
pip install diffusers>=0.27.0
pip install huggingface-hub>=0.20.0
pip install accelerate>=0.24.0
pip install safetensors>=0.4.0
pip install invisible-watermark>=0.2.0
```

## 📋 필요한 최소 버전

- **transformers**: >= 4.40.0 (CLIPImageProcessor 지원)
- **diffusers**: >= 0.27.0 (최신 기능 및 호환성)
- **huggingface-hub**: >= 0.20.0
- **accelerate**: >= 0.24.0
- **safetensors**: >= 0.4.0
- **invisible-watermark**: >= 0.2.0 (SDXL 필수)

## ⚠️ xformers 경고

다음과 같은 경고는 무시해도 됩니다:

```
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions...
```

이것은:
- CPU 버전 PyTorch를 사용하는 경우 정상입니다
- GPU가 없거나 CUDA가 설치되지 않은 경우 발생합니다
- xformers는 GPU에서만 필요하며, CPU에서는 사용되지 않습니다
- 프로그램 실행에는 영향을 주지 않습니다

## 🔍 버전 확인

설치된 버전을 확인하려면:

```bash
pip show transformers diffusers
```

또는 Python에서:

```python
import transformers
import diffusers
print(f"transformers: {transformers.__version__}")
print(f"diffusers: {diffusers.__version__}")
```

## 🐛 여전히 문제가 발생하는 경우

1. **가상 환경 사용 확인**
   ```bash
   # 가상 환경 활성화
   conda activate aivideo
   # 또는
   source venv/bin/activate
   ```

2. **캐시 정리**
   ```bash
   pip cache purge
   ```

3. **Python 버전 확인**
   - Python 3.10 이상 필요
   ```bash
   python --version
   ```

4. **전체 재설치**
   ```bash
   pip install -r requirements.txt --force-reinstall --no-cache-dir
   ```

## 📞 추가 도움

문제가 계속되면:
1. 에러 메시지 전체를 확인
2. `pip list` 출력 확인
3. Python 버전 확인
4. 가상 환경 사용 여부 확인
