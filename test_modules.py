"""모듈 테스트 스크립트"""

import sys
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("=" * 60)
print("모듈 테스트 시작")
print("=" * 60)

# 1. 모듈 import 테스트
print("\n[1/6] 모듈 import 테스트...")
try:
    from src import audio_extract, stt_whisper, translate_llm, tts_korean, subtitles, renderer, pipeline
    print("✅ 모든 모듈 import 성공")
except Exception as e:
    print(f"❌ 모듈 import 실패: {e}")
    sys.exit(1)

# 2. 설정 파일 로드 테스트
print("\n[2/6] 설정 파일 로드 테스트...")
try:
    import yaml
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("✅ 설정 파일 로드 성공")
except Exception as e:
    print(f"❌ 설정 파일 로드 실패: {e}")
    sys.exit(1)

# 3. Whisper STT 초기화 테스트
print("\n[3/6] Whisper STT 초기화 테스트...")
try:
    whisper_stt = stt_whisper.create_whisper_stt(config)
    print("✅ Whisper STT 초기화 성공 (모델은 아직 로드하지 않음)")
except Exception as e:
    print(f"❌ Whisper STT 초기화 실패: {e}")

# 4. 자막 생성기 초기화 테스트
print("\n[4/6] 자막 생성기 초기화 테스트...")
try:
    subtitle_gen = subtitles.create_subtitle_generator(config)
    print("✅ 자막 생성기 초기화 성공")
except Exception as e:
    print(f"❌ 자막 생성기 초기화 실패: {e}")

# 5. 비디오 렌더러 초기화 테스트
print("\n[5/6] 비디오 렌더러 초기화 테스트...")
try:
    renderer_obj = renderer.create_video_renderer(config)
    print("✅ 비디오 렌더러 초기화 성공")
except Exception as e:
    print(f"❌ 비디오 렌더러 초기화 실패: {e}")

# 6. 파이프라인 초기화 테스트
print("\n[6/6] 파이프라인 초기화 테스트...")
try:
    pipeline_obj = pipeline.VideoConversionPipeline(config=config)
    print("✅ 파이프라인 초기화 성공")
except Exception as e:
    print(f"❌ 파이프라인 초기화 실패: {e}")

print("\n" + "=" * 60)
print("모듈 테스트 완료!")
print("=" * 60)
print("\n다음 단계:")
print("1. FFmpeg 설치 확인 (필수)")
print("2. 테스트 비디오 파일 준비")
print("3. 다음 명령어로 실행:")
print("   python run_pipeline.py --input video.mp4 --output output.mp4")
print("\n주의: LLM 모델과 TTS 모델은 첫 실행 시 자동으로 다운로드됩니다.")
print("      이는 수 GB의 용량과 시간이 필요할 수 있습니다.")

