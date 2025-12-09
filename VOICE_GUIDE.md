# 🎤 목소리 선택 가이드

## 🎬 현재 기본 설정: 영화 나레이션 스타일

프로그램이 기본으로 사용하는 목소리:
- **edge-tts**: `en-US-DavisNeural` (남성, 영화/다큐멘터리 나레이션)
- **Coqui TTS**: `p245` 화자 (남성, 깊고 진지한 목소리)

> 🎥 **마블 영화나 BBC 다큐멘터리 같은 진지하고 권위있는 할아버지 나레이션 스타일입니다!**

---

## 🎭 목소리 스타일별 추천

### 🎬 영화/다큐멘터리 나레이션 (현재 기본)

**느낌:** 마블 영화 예고편, BBC 다큐멘터리, 넷플릭스 시리즈

**edge-tts:**
```
en-US-DavisNeural    ⭐ (현재 기본) - 영화 예고편 같은 진지한 목소리
en-US-TonyNeural     - 뉴스 앵커 같은 전문적인 목소리
en-GB-RyanNeural     - 영국 다큐멘터리 스타일
```

**Coqui TTS:**
```
speaker="p245"  ⭐ (현재 기본) - 깊고 진지한 남성 목소리
speaker="p232"  - 나이든 목소리 (할아버지 느낌)
speaker="p243"  - 중년 남성, 권위있는 목소리
```

---

### 👨 자연스러운 일반 남성 목소리

**느낌:** 친근하고 편안한 대화체

**edge-tts:**
```
en-US-GuyNeural      - 자연스럽고 친근한 목소리
en-US-JasonNeural    - 젊고 활기찬 목소리
en-US-ChristopherNeural - 부드럽고 따뜻한 목소리
```

---

### 👩 여성 목소리

**느낌:** 전문적이거나 친근한 여성 목소리

**edge-tts:**
```
en-US-AriaNeural     - 자연스럽고 전문적
en-US-JennyNeural    - 친근하고 따뜻함
en-US-SaraNeural     - 차분하고 신뢰감 있음
en-US-MichelleNeural - 활기차고 에너제틱
```

---

### 🎙️ 특수 스타일

**프레젠테이션/강의:**
```
en-US-BrandonNeural  - 강의 스타일
en-US-RogerNeural    - 연설가 스타일
```

**스토리텔링:**
```
en-US-EricNeural     - 이야기꾼 스타일
en-US-SteffanNeural  - 오디오북 낭독 스타일
```

**영국 억양:**
```
en-GB-RyanNeural     - 영국 다큐멘터리 (신사 느낌)
en-GB-ThomasNeural   - 영국 뉴스 앵커
```

---

## 🔧 목소리 변경 방법

### 방법 1: 코드 직접 수정

`src/video_generator.py` 파일에서:

#### edge-tts 목소리 변경:
```python
# 1470번째 줄 근처
communicate = edge_tts.Communicate(clean_text, "en-US-DavisNeural")

# 원하는 목소리로 변경:
communicate = edge_tts.Communicate(clean_text, "en-US-GuyNeural")
```

#### Coqui TTS 화자 변경:
```python
# 1450번째 줄 근처
tts_model.tts_to_file(text=clean_text, file_path=output_path, speaker="p245")

# 원하는 화자로 변경:
tts_model.tts_to_file(text=clean_text, file_path=output_path, speaker="p232")
```

---

## 🎧 목소리 미리듣기

### edge-tts로 테스트:

```powershell
# DavisNeural (현재 기본) 테스트
edge-tts --voice en-US-DavisNeural --text "Welcome to our comprehensive demonstration of AI-powered video creation." --write-media test_davis.mp3

# GuyNeural 테스트
edge-tts --voice en-US-GuyNeural --text "Welcome to our comprehensive demonstration of AI-powered video creation." --write-media test_guy.mp3

# AriaNeural (여성) 테스트
edge-tts --voice en-US-AriaNeural --text "Welcome to our comprehensive demonstration of AI-powered video creation." --write-media test_aria.mp3
```

### Coqui TTS로 테스트:

```powershell
python -c "from TTS.api import TTS; tts = TTS(model_name='tts_models/en/vctk/vits', progress_bar=False, gpu=False); tts.tts_to_file(text='Welcome to our comprehensive demonstration', file_path='test_p245.wav', speaker='p245')"

python -c "from TTS.api import TTS; tts = TTS(model_name='tts_models/en/vctk/vits', progress_bar=False, gpu=False); tts.tts_to_file(text='Welcome to our comprehensive demonstration', file_path='test_p232.wav', speaker='p232')"
```

---

## 📋 전체 목소리 목록 확인

### edge-tts 목소리 전체 목록:

```powershell
# 모든 영어 목소리 확인
edge-tts --list-voices | findstr "en-US"

# 남성 목소리만 확인
edge-tts --list-voices | findstr "en-US.*Male"

# 여성 목소리만 확인
edge-tts --list-voices | findstr "en-US.*Female"
```

### Coqui TTS 화자 목록:

```powershell
python -c "from TTS.api import TTS; tts = TTS(model_name='tts_models/en/vctk/vits'); print(tts.speakers)"
```

---

## 🎯 추천 조합

### 🏆 영화 예고편 스타일 (현재 기본):
```
edge-tts: en-US-DavisNeural
Coqui: speaker="p245"
```
**느낌:** "In a world..." 🎬

### 📺 다큐멘터리 스타일:
```
edge-tts: en-GB-RyanNeural
Coqui: speaker="p232"
```
**느낌:** BBC 자연 다큐멘터리

### 💼 비즈니스 프레젠테이션:
```
edge-tts: en-US-TonyNeural
Coqui: speaker="p243"
```
**느낌:** 전문적이고 신뢰감 있는 목소리

### 🎓 교육 콘텐츠:
```
edge-tts: en-US-BrandonNeural
Coqui: speaker="p225"
```
**느낌:** 친근하고 이해하기 쉬운 설명

### 📚 오디오북:
```
edge-tts: en-US-EricNeural
Coqui: speaker="p260"
```
**느낌:** 스토리텔링에 적합

---

## ⚙️ 고급 설정

### 목소리 속도 조절 (edge-tts):

```python
# 느리게
communicate = edge_tts.Communicate(clean_text, "en-US-DavisNeural", rate="-20%")

# 빠르게
communicate = edge_tts.Communicate(clean_text, "en-US-DavisNeural", rate="+20%")
```

### 음높이 조절 (edge-tts):

```python
# 낮게 (더 남성적)
communicate = edge_tts.Communicate(clean_text, "en-US-DavisNeural", pitch="-10Hz")

# 높게
communicate = edge_tts.Communicate(clean_text, "en-US-DavisNeural", pitch="+10Hz")
```

---

## 💡 팁

1. **영화 같은 느낌**: `en-US-DavisNeural` (현재 기본) 사용
2. **친근한 느낌**: `en-US-GuyNeural` 사용
3. **전문적인 느낌**: `en-US-TonyNeural` 사용
4. **영국 신사**: `en-GB-RyanNeural` 사용
5. **할아버지 느낌**: Coqui `speaker="p232"` 사용

---

## 🎬 마블/영화 스타일 최적화

현재 기본 설정이 이미 마블 예고편 같은 스타일이지만, 더 조정하고 싶다면:

```python
# 더 낮고 진지하게
communicate = edge_tts.Communicate(
    clean_text, 
    "en-US-DavisNeural",
    rate="-10%",  # 약간 느리게
    pitch="-5Hz"  # 약간 낮게
)
```

---

## ✅ 테스트 체크리스트

목소리를 변경한 후:

- [ ] 코드 저장
- [ ] 프로그램 재실행: `python main.py`
- [ ] 생성된 비디오 확인
- [ ] 목소리가 마음에 들면 완료!
- [ ] 마음에 안 들면 다른 목소리로 변경 후 재시도

---

## 🆘 도움말

- 목소리가 이상하면: 다른 목소리 시도
- 너무 빠르거나 느리면: `rate` 조절
- 너무 높거나 낮으면: `pitch` 조절
- 영화 같은 느낌 원하면: `en-US-DavisNeural` 유지 (현재 기본)
