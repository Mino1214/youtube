# TTS (Text-to-Speech) 문제 해결 가이드

## 🎬 현재 설정: 영화 나레이션 스타일 남성 목소리

프로그램이 기본으로 사용하는 목소리:
- **edge-tts**: `en-US-DavisNeural` (남성, 영화/다큐멘터리 나레이션)
- **Coqui TTS**: `p245` 화자 (남성, 깊고 진지한 목소리)

> 💡 **마블 영화나 다큐멘터리 같은 진지한 할아버지 나레이션 스타일입니다!**

---

## 문제 상황

"오디오 파일이 비어있습니다 (0 bytes)" 오류가 발생하는 경우

---

## ✅ 즉시 해결 방법 (고품질 TTS 엔진 설치)

Piper TTS가 작동하지 않을 때, 프로그램이 자동으로 다음 엔진들을 시도합니다:

### 🏆 옵션 1: Coqui TTS - 최고 품질 (강력 권장!) ⭐⭐⭐

**장점:**
- 🎤 **가장 자연스러운 음성** (사람 목소리에 가까움)
- 오픈소스, 무료
- 오프라인 사용 가능
- 다양한 고품질 음성 모델

**단점:**
- 설치 크기가 큼 (~500MB)
- 첫 실행 시 모델 다운로드 시간 필요
- CPU 사용 시 약간 느림

**설치:**
```powershell
pip install TTS
```

**테스트:**
```powershell
python -c "from TTS.api import TTS; tts = TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC', progress_bar=False, gpu=False); tts.tts_to_file(text='Hello world', file_path='test.wav'); print('Success!')"
```

**추천 이유:** 가장 자연스럽고 고품질이며, 오프라인 사용 가능!

---

### 🥈 옵션 2: edge-tts (Microsoft Edge TTS) - 매우 자연스러움 ⭐⭐

**장점:**
- 🎤 **매우 자연스러운 음성** (신경망 기반)
- Microsoft 품질
- 무료, 빠름
- 다양한 목소리 선택 가능

**단점:**
- 인터넷 연결 필수
- Microsoft API 의존

**설치:**
```powershell
pip install edge-tts
```

**테스트:**
```powershell
python -c "import asyncio; import edge_tts; asyncio.run(edge_tts.Communicate('Hello world', 'en-US-AriaNeural').save('test.mp3')); print('Success!')"
```

**추천 이유:** 빠르고 매우 자연스러우며 설치가 간단!

---

### 🥉 옵션 3: gTTS (Google Text-to-Speech) - 간단하고 빠름⭐

**장점:**
- 설치 매우 간단
- 빠른 처리 속도
- 안정적

**단점:**
- 인터넷 연결 필요
- 음성이 약간 로봇틱

**설치:**
```powershell
pip install gtts
```

**테스트:**
```powershell
python -c "from gtts import gTTS; tts = gTTS(text='Hello world', lang='en'); tts.save('test.mp3'); print('Success!')"
```

**추천 이유:** 가장 간단하고 빠르게 작동!

---

### 옵션 3: 무음 오디오 (기본 fallback)

모든 TTS 엔진이 실패하면, 프로그램은 자동으로 **무음 오디오**를 생성합니다.

---

## 🔧 Piper TTS 수리 (고급)

Piper TTS를 계속 사용하고 싶다면:

### 1. Piper 재설치

```powershell
pip uninstall piper-tts -y
pip install piper-tts
```

### 2. 모델 재다운로드

```powershell
python download_all_models.py --auto
```

### 3. 모델 파일 확인

```powershell
# 모델 디렉토리 확인
dir C:\Users\%USERNAME%\.local\share\piper\voices\en\en_US\amy\medium
```

파일이 있어야 합니다:
- `en_US-amy-medium.onnx` (모델 파일)
- `en_US-amy-medium.onnx.json` (설정 파일)

### 4. 모델 테스트

```powershell
python -c "from piper import PiperVoice; model = PiperVoice.load('C:/Users/YOUR_USERNAME/.local/share/piper/voices/en/en_US/amy/medium/en_US-amy-medium.onnx'); print('Model loaded successfully!')"
```

---

## 📊 프로그램 동작 방식

프로그램이 영어 TTS를 생성할 때 다음 순서로 시도합니다:

```
1️⃣ Piper TTS 시도 (오프라인, 기본)
   ↓ 실패 시
2️⃣ Coqui TTS 시도 (최고 품질, 오프라인) 🏆
   ↓ 실패 시
3️⃣ edge-tts 시도 (매우 자연스러움, 온라인) 
   ↓ 실패 시
4️⃣ gTTS 시도 (간단하고 빠름, 온라인)
   ↓ 실패 시
5️⃣ 무음 오디오 생성 (최종 fallback)
```

---

## 🎯 품질 비교표

| TTS 엔진 | 자연스러움 | 속도 | 인터넷 | 크기 | 추천도 |
|---------|----------|------|--------|------|--------|
| **Coqui TTS** 🏆 | ⭐⭐⭐⭐⭐ | 중간 | ❌ 불필요 | 대형 | ⭐⭐⭐⭐⭐ |
| **edge-tts** | ⭐⭐⭐⭐⭐ | 빠름 | ✅ 필요 | 소형 | ⭐⭐⭐⭐ |
| **gTTS** | ⭐⭐⭐ | 매우 빠름 | ✅ 필요 | 소형 | ⭐⭐⭐ |
| Piper | ⭐⭐⭐ | 빠름 | ❌ 불필요 | 중형 | ⭐⭐ |
| 무음 | - | 즉시 | ❌ 불필요 | - | - |

---

## ⚡ 빠른 해결책

### 🏆 최고 품질을 원한다면 (강력 권장!):

```powershell
# Coqui TTS 설치 - 가장 자연스러운 음성!
pip install TTS

# 프로그램 다시 실행
python main.py
```

### 🚀 빠르고 자연스러운 음성을 원한다면:

```powershell
# edge-tts 설치 - 매우 자연스럽고 빠름!
pip install edge-tts

# 프로그램 다시 실행
python main.py
```

### ⚡ 가장 간단한 방법:

```powershell
# gTTS 설치 - 간단하고 빠름
pip install gtts

# 프로그램 다시 실행
python main.py
```

이제 **고품질 실제 음성**이 생성됩니다! 🎉

---

## 🔍 로그 확인

프로그램 실행 시 다음과 같은 로그를 확인하세요:

### ✅ 성공 (Coqui TTS - 최고 품질):
```
🎙️  Coqui TTS (고품질, 자연스러움) 시도 중...
✅ Coqui TTS로 고품질 오디오 생성 완료: path.wav (234567 bytes)
```

### ✅ 성공 (edge-tts - 매우 자연스러움):
```
🎙️  edge-tts (Microsoft Edge TTS, 매우 자연스러움) 시도 중...
✅ edge-tts로 고품질 오디오 생성 완료: path.wav (234567 bytes)
```

### ✅ 성공 (gTTS - 빠르고 간단):
```
🎙️  gTTS (Google Text-to-Speech) 시도 중...
✅ gTTS로 오디오 생성 완료: path.wav (123456 bytes)
```

### ⚠️ 무음 오디오 사용:
```
모든 TTS 엔진이 실패했습니다. 무음 오디오로 대체합니다...
무음 오디오 생성 완료: path.wav (길이: 10.0초)
```

---

## 💡 추천 설정

### 🏆 최고 품질 설정 (오프라인 사용 가능):
```powershell
# Coqui TTS - 가장 자연스러운 음성!
pip install TTS

# 첫 실행 시 자동으로 모델 다운로드됨 (~500MB)
```

### 🚀 균형잡힌 설정 (온라인/오프라인 모두):
```powershell
# Coqui TTS (오프라인, 최고 품질) + edge-tts (온라인, 빠름)
pip install TTS edge-tts

# Coqui가 1순위, edge-tts가 2순위로 작동
```

### ⚡ 간단한 설정 (인터넷 연결 필요):
```powershell
# edge-tts + gTTS (둘 다 빠르고 간단)
pip install edge-tts gtts
```

### 📱 오프라인 전용 설정:
```powershell
# Coqui TTS만 설치
pip install TTS

# 또는 Piper 모델을 수리
python download_all_models.py --auto
```

---

## 🆘 추가 도움말

### Q: 가장 자연스러운 목소리를 원해요!
A: **Coqui TTS를 설치하세요!** 🏆
```powershell
pip install TTS
```
가장 사람 같은 목소리를 제공합니다.

### Q: 인터넷이 없으면 어떻게 하나요?
A: **Coqui TTS**를 사용하세요 (오프라인 작동). 또는 Piper를 수리하세요.

### Q: Coqui TTS가 너무 느려요
A: 다음 옵션들을 시도하세요:
- **edge-tts** (온라인, 빠르고 자연스러움)
- **gTTS** (온라인, 매우 빠름)

### Q: 설치 크기가 부담스러워요
A: 
- **Coqui TTS**: ~500MB (최고 품질)
- **edge-tts**: ~50KB (매우 작음)
- **gTTS**: ~100KB (매우 작음)

### Q: 모든 방법이 실패했어요
A: 무음 오디오가 자동으로 생성됩니다. 비디오는 정상적으로 생성되지만 소리가 없습니다. 자막이 있어서 내용은 이해 가능합니다.

### Q: 다른 목소리를 사용하고 싶어요
A: **현재 기본 설정: 영화 나레이션 스타일 남성 목소리** 🎬

프로그램이 사용하는 목소리:
- **edge-tts**: `en-US-DavisNeural` (남성, 영화/다큐멘터리 나레이션 스타일)
- **Coqui TTS**: `p245` 화자 (남성, 깊고 진지한 목소리)

다른 목소리로 변경하려면:

#### 🎬 영화 나레이션 스타일 (현재 기본):
```python
# edge-tts
"en-US-DavisNeural"  # 남성, 다큐멘터리/영화 나레이션 (마블 같은 느낌)

# Coqui TTS
speaker="p245"  # 남성, 깊고 진지한 목소리
speaker="p232"  # 남성, 나이든 목소리 (할아버지 느낌)
```

#### 👨 일반 남성 목소리:
```python
# edge-tts
"en-US-GuyNeural"    # 남성, 자연스러운 일반 목소리
"en-US-TonyNeural"   # 남성, 뉴스캐스터 스타일
"en-US-JasonNeural"  # 남성, 젊은 목소리
```

#### 👩 여성 목소리:
```python
# edge-tts
"en-US-AriaNeural"   # 여성, 자연스러운 목소리
"en-US-JennyNeural"  # 여성, 친근한 목소리
"en-US-SaraNeural"   # 여성, 전문적인 목소리
```

#### 🌍 다른 억양:
```python
# gTTS
gTTS(text='...', lang='en', tld='co.uk')   # 영국 영어 (신사 느낌)
gTTS(text='...', lang='en', tld='com.au')  # 호주 영어
gTTS(text='...', lang='en', tld='ca')      # 캐나다 영어
```

#### 📜 목소리 전체 목록 확인:
```powershell
# edge-tts 모든 목소리 확인
edge-tts --list-voices | findstr "en-US"
```

### Q: GPU를 사용해서 더 빠르게 할 수 있나요?
A: 네! Coqui TTS는 GPU를 지원합니다:
```python
TTS(model_name='...', gpu=True)  # GPU 사용
```
하지만 현재 코드에서는 호환성을 위해 CPU 모드(gpu=False)를 사용합니다.

---

## 📞 관련 링크

- [gTTS 공식 문서](https://gtts.readthedocs.io/)
- [edge-tts GitHub](https://github.com/rany2/edge-tts)
- [Piper TTS GitHub](https://github.com/rhasspy/piper)

---

## ✅ 체크리스트

프로그램을 다시 실행하기 전에:

### 🏆 최고 품질 (강력 권장!):
- [ ] `pip install TTS` 실행 (Coqui TTS, 최고 품질)
- [ ] 프로그램 재실행: `python main.py`
- [ ] 첫 실행 시 모델 다운로드 대기 (약 1-2분)

### 🚀 빠르고 자연스러운:
- [ ] `pip install edge-tts` 실행 (매우 자연스러움)
- [ ] 인터넷 연결 확인
- [ ] 프로그램 재실행: `python main.py`

### ⚡ 간단하고 빠른:
- [ ] `pip install gtts` 실행 (가장 간단)
- [ ] 인터넷 연결 확인
- [ ] 프로그램 재실행: `python main.py`

### 📱 모든 옵션 설치 (최고의 fallback):
- [ ] `pip install TTS edge-tts gtts` 실행
- [ ] 프로그램 재실행: `python main.py`
- [ ] Coqui → edge-tts → gTTS 순서로 자동 시도

---

## 🎯 권장 명령어

**가장 추천하는 방법 (최고 품질):**
```powershell
pip install TTS
python main.py
```

**빠르게 테스트하고 싶다면:**
```powershell
pip install edge-tts
python main.py
```

**가장 간단한 방법:**
```powershell
pip install gtts
python main.py
```
