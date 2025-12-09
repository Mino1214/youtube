"""전체 파이프라인 조율 모듈"""

import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from .audio_extract import process_input
from .stt_whisper import create_whisper_stt
from .translate_llm import create_llm_translator
from .tts_korean import create_korean_tts
from .subtitles import create_subtitle_generator
from .renderer import create_video_renderer
from .video_generator import create_video_generator, create_english_tts

logger = logging.getLogger(__name__)


class VideoConversionPipeline:
    """비디오 변환 파이프라인"""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[dict] = None, 
                 channel: Optional[str] = None):
        """
        Args:
            config_path: 설정 파일 경로
            config: 설정 딕셔너리 (config_path보다 우선)
            channel: 채널 프로필 이름 (예: "narration_shorts", "short_story")
        """
        if config is None:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
            else:
                # config_path가 없으면 기본 경로에서 config.yaml 찾기
                default_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
                if os.path.exists(default_config_path):
                    logger.info(f"기본 설정 파일 사용: {default_config_path}")
                    with open(default_config_path, 'r', encoding='utf-8') as f:
                        self.config = yaml.safe_load(f)
                else:
                    # 기본 설정
                    logger.warning("config.yaml 파일을 찾을 수 없습니다. 기본 설정을 사용합니다.")
                    self.config = self._get_default_config()
        else:
            self.config = config
        
        # 채널 프로필 로드 및 적용
        self.channel_profile = None
        if channel:
            self.channel_profile = self._load_channel_profile(channel)
            if self.channel_profile:
                self._apply_channel_profile()
                logger.info(f"채널 프로필 적용: {self.channel_profile.get('name', channel)}")
        
        # 임시 디렉토리 설정
        temp_config = self.config.get("temp", {})
        self.temp_dir = temp_config.get("temp_dir")
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="aivideo_")
        else:
            os.makedirs(self.temp_dir, exist_ok=True)
        
        self.auto_cleanup = temp_config.get("auto_cleanup", True)
        self.temp_files = []  # 정리할 임시 파일 목록
        
        # 컴포넌트 초기화
        self.whisper_stt = None
        self.llm_translator = None
        self.korean_tts = None
        self.subtitle_generator = None
        self.video_renderer = None
        self.video_generator = None
        self.english_tts = None
        
        logger.info(f"파이프라인 초기화 완료 (임시 디렉토리: {self.temp_dir})")
    
    def _load_channel_profile(self, channel_name: str) -> Optional[dict]:
        """채널 프로필 로드"""
        try:
            # channels.yaml 파일 경로 찾기
            project_root = os.path.dirname(os.path.dirname(__file__))
            channels_path = os.path.join(project_root, "channels.yaml")
            
            if not os.path.exists(channels_path):
                logger.warning(f"channels.yaml 파일을 찾을 수 없습니다: {channels_path}")
                return None
            
            with open(channels_path, 'r', encoding='utf-8') as f:
                channels_config = yaml.safe_load(f)
            
            channels = channels_config.get("channels", {})
            if channel_name not in channels:
                logger.warning(f"채널 '{channel_name}'을 찾을 수 없습니다. 사용 가능한 채널: {list(channels.keys())}")
                # 기본 채널 사용
                default_channel = channels_config.get("default_channel", "narration_shorts")
                if default_channel in channels:
                    logger.info(f"기본 채널 사용: {default_channel}")
                    channel_name = default_channel
                else:
                    return None
            
            profile = channels[channel_name]
            profile["_channel_id"] = channel_name  # 내부 사용
            return profile
            
        except Exception as e:
            logger.error(f"채널 프로필 로드 실패: {e}")
            return None
    
    def _apply_channel_profile(self):
        """채널 프로필을 설정에 적용"""
        if not self.channel_profile:
            return
        
        # TTS 설정 적용
        if "tts" in self.channel_profile:
            channel_tts = self.channel_profile["tts"]
            if "piper" not in self.config.get("tts", {}):
                self.config.setdefault("tts", {})["piper"] = {}
            
            if "voice" in channel_tts:
                self.config["tts"]["piper"]["voice"] = channel_tts["voice"]
            if "speed" in channel_tts:
                self.config["tts"]["piper"]["speed"] = channel_tts["speed"]
        
        # 자막 설정 적용
        if "subtitles" in self.channel_profile:
            channel_subs = self.channel_profile["subtitles"]
            if "style" not in self.config.get("subtitles", {}):
                self.config.setdefault("subtitles", {})["style"] = {}
            
            if "font_size" in channel_subs:
                self.config["subtitles"]["style"]["font_size"] = channel_subs["font_size"]
            if "position" in channel_subs:
                self.config["subtitles"]["style"]["position"] = channel_subs["position"]
        
        # 비디오 생성 설정 적용
        if "video" in self.channel_profile:
            channel_video = self.channel_profile["video"]
            if "video_generation" not in self.config:
                self.config["video_generation"] = {}
            
            if "background_color" in channel_video:
                self.config["video_generation"]["background_color"] = channel_video["background_color"]
            if "fps" in channel_video:
                self.config["video_generation"]["fps"] = channel_video["fps"]
            if "max_duration" in channel_video:
                self.config["video_generation"]["max_duration"] = channel_video["max_duration"]
        
        logger.debug("채널 프로필이 설정에 적용되었습니다.")
    
    def _get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            "whisper": {
                "model": "large-v3",
                "device": "cuda",
                "language": "en",
                "word_timestamps": True
            },
            "llm": {
                "model": "deepseek-r1-7b",
                "use_gpu": True,
                "batch_size": 4,
                "max_tokens": 2048
            },
            "tts": {
                "model": "piper",
                "piper": {
                    "voice": "ko_KR-hyeri-medium",
                    "speed": 1.0,
                    "noise_scale": 0.667
                }
            },
            "subtitles": {
                "mode": "korean_only"
            },
            "video": {
                "video_codec": "libx264",
                "audio_codec": "aac",
                "video_bitrate": "5000k",
                "audio_bitrate": "192k"
            },
            "temp": {
                "auto_cleanup": True
            }
        }
    
    def _register_temp_file(self, file_path: str):
        """임시 파일 등록"""
        self.temp_files.append(file_path)
    
    def cleanup(self):
        """임시 파일 정리"""
        if not self.auto_cleanup:
            return
        
        logger.info("임시 파일 정리 중...")
        
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"삭제됨: {file_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패 ({file_path}): {e}")
        
        # 임시 디렉토리 삭제
        try:
            if os.path.exists(self.temp_dir) and self.temp_dir.startswith(tempfile.gettempdir()):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug(f"임시 디렉토리 삭제됨: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"임시 디렉토리 삭제 실패: {e}")
    
    def run(self, input_path: str, output_path: str) -> str:
        """
        전체 파이프라인 실행
        
        Args:
            input_path: 입력 비디오 파일 또는 URL
            output_path: 출력 비디오 파일 경로
            
        Returns:
            생성된 비디오 파일 경로
        """
        try:
            logger.info("=" * 60)
            logger.info("비디오 변환 파이프라인 시작")
            logger.info("=" * 60)
            
            # 1. 오디오 추출
            logger.info("\n[1/7] 오디오 추출 중...")
            video_path, audio_path = process_input(input_path, self.temp_dir)
            self._register_temp_file(audio_path)
            if video_path != input_path and not os.path.abspath(video_path) == os.path.abspath(input_path):
                self._register_temp_file(video_path)
            
            # 2. STT 처리
            logger.info("\n[2/7] Speech-to-Text 처리 중...")
            if self.whisper_stt is None:
                self.whisper_stt = create_whisper_stt(self.config)
            stt_result = self.whisper_stt.transcribe(audio_path)
            segments = stt_result["segments"]
            logger.info(f"STT 완료: {len(segments)}개 세그먼트, {len(stt_result['text'])}자")
            
            # 3. 번역
            logger.info("\n[3/7] 영어→한국어 번역 중...")
            if self.llm_translator is None:
                self.llm_translator = create_llm_translator(self.config)
            # 채널 프로필의 번역 스타일 전달
            translation_style = self.channel_profile.get("translation_style", {}) if self.channel_profile else {}
            translated_segments = self.llm_translator.translate_segments(segments, translation_style=translation_style)
            logger.info(f"번역 완료: {len(translated_segments)}개 세그먼트")
            
            # 4. TTS 생성
            logger.info("\n[4/7] 한국어 TTS 생성 중...")
            if self.korean_tts is None:
                self.korean_tts = create_korean_tts(self.config)
            
            # 전체 텍스트를 하나의 오디오로 생성
            # (세그먼트별 생성은 복잡하므로 일단 전체 텍스트로 생성)
            full_korean_text = " ".join([seg.get("text_ko", "") for seg in translated_segments if seg.get("text_ko")])
            if not full_korean_text:
                raise ValueError("번역된 텍스트가 없습니다.")
            
            narration_audio_path = os.path.join(self.temp_dir, "narration.wav")
            self.korean_tts.synthesize(full_korean_text, narration_audio_path)
            self._register_temp_file(narration_audio_path)
            logger.info("TTS 생성 완료")
            
            # 5. 자막 생성
            logger.info("\n[5/7] 자막 파일 생성 중...")
            if self.subtitle_generator is None:
                self.subtitle_generator = create_subtitle_generator(self.config)
            subtitle_path = os.path.join(self.temp_dir, "subtitles.srt")
            self.subtitle_generator.generate_srt(translated_segments, subtitle_path)
            self._register_temp_file(subtitle_path)
            logger.info(f"자막 생성 완료: {subtitle_path}")
            
            # 6. 원본 비디오 음소거
            logger.info("\n[6/7] 원본 비디오 음소거 중...")
            if self.video_renderer is None:
                self.video_renderer = create_video_renderer(self.config)
            muted_video_path = os.path.join(self.temp_dir, "muted_video.mp4")
            self.video_renderer.mute_original_audio(video_path, muted_video_path)
            self._register_temp_file(muted_video_path)
            
            # 7. 최종 비디오 렌더링
            logger.info("\n[7/7] 최종 비디오 렌더링 중...")
            final_video_path = self.video_renderer.render(
                muted_video_path,
                narration_audio_path,
                subtitle_path,
                output_path
            )
            
            logger.info("=" * 60)
            logger.info("파이프라인 완료!")
            logger.info(f"출력 파일: {final_video_path}")
            logger.info("=" * 60)
            
            return final_video_path
            
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)
            raise
        finally:
            # 임시 파일 정리
            if self.auto_cleanup:
                self.cleanup()
    
    def _get_output_path(self, default_path: str) -> str:
        """채널 프로필에 따라 출력 경로 생성"""
        if not self.channel_profile:
            return default_path
        
        output_config = self.channel_profile.get("output", {})
        output_dir = output_config.get("directory")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # 파일명 패턴 처리
            filename_pattern = output_config.get("filename_pattern", "{timestamp}_{title}.mp4")
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            title = os.path.splitext(os.path.basename(default_path))[0]
            filename = filename_pattern.format(timestamp=timestamp, title=title)
            return os.path.join(output_dir, filename)
        
        return default_path
    
    def run_from_text(self, english_text: str, output_path: str, channel: Optional[str] = None) -> str:
        """
        영어 텍스트로부터 한국어 비디오 생성 (전체 파이프라인)
        
        Args:
            english_text: 영어 텍스트/스크립트
            output_path: 출력 비디오 파일 경로
            channel: 채널 프로필 이름 (None이면 초기화 시 설정된 채널 사용)
            
        Returns:
            생성된 비디오 파일 경로
        """
        # 채널이 런타임에 지정된 경우 프로필 로드
        if channel and channel != getattr(self, '_current_channel', None):
            self.channel_profile = self._load_channel_profile(channel)
            if self.channel_profile:
                self._apply_channel_profile()
                self._current_channel = channel
        
        # 출력 경로 조정
        output_path = self._get_output_path(output_path)
        
        try:
            logger.info("=" * 60)
            logger.info("텍스트→비디오 파이프라인 시작")
            if self.channel_profile:
                logger.info(f"채널: {self.channel_profile.get('name', 'Unknown')}")
            logger.info("=" * 60)
            
            # 1. 영어 TTS 생성
            logger.info("\n[1/8] 영어 TTS 생성 중...")
            if self.english_tts is None:
                self.english_tts = create_english_tts(self.config)
            english_audio_path = os.path.join(self.temp_dir, "english_audio.wav")
            self.english_tts.synthesize(english_text, english_audio_path)
            self._register_temp_file(english_audio_path)
            logger.info("영어 TTS 생성 완료")
            
            # 2. 영어 비디오 생성
            logger.info("\n[2/8] 영어 비디오 생성 중...")
            if self.video_generator is None:
                self.video_generator = create_video_generator(self.config)
            english_video_path = os.path.join(self.temp_dir, "english_video.mp4")
            self.video_generator.generate_from_text(
                english_text, 
                english_audio_path, 
                english_video_path
            )
            self._register_temp_file(english_video_path)
            logger.info("영어 비디오 생성 완료")
            
            # 3. STT 처리 (생성된 비디오에서)
            logger.info("\n[3/8] Speech-to-Text 처리 중...")
            if self.whisper_stt is None:
                self.whisper_stt = create_whisper_stt(self.config)
            stt_result = self.whisper_stt.transcribe(english_audio_path)
            segments = stt_result["segments"]
            logger.info(f"STT 완료: {len(segments)}개 세그먼트")
            
            # 4. 번역
            logger.info("\n[4/8] 영어→한국어 번역 중...")
            if self.llm_translator is None:
                self.llm_translator = create_llm_translator(self.config)
            # 채널 프로필의 번역 스타일 전달
            translation_style = self.channel_profile.get("translation_style", {}) if self.channel_profile else {}
            translated_segments = self.llm_translator.translate_segments(segments, translation_style=translation_style)
            logger.info(f"번역 완료: {len(translated_segments)}개 세그먼트")
            
            # 5. 한국어 TTS 생성
            logger.info("\n[5/8] 한국어 TTS 생성 중...")
            if self.korean_tts is None:
                self.korean_tts = create_korean_tts(self.config)
            full_korean_text = " ".join([seg.get("text_ko", "") for seg in translated_segments if seg.get("text_ko")])
            if not full_korean_text:
                raise ValueError("번역된 텍스트가 없습니다.")
            narration_audio_path = os.path.join(self.temp_dir, "narration.wav")
            self.korean_tts.synthesize(full_korean_text, narration_audio_path)
            self._register_temp_file(narration_audio_path)
            logger.info("한국어 TTS 생성 완료")
            
            # 6. 자막 생성
            logger.info("\n[6/8] 자막 파일 생성 중...")
            if self.subtitle_generator is None:
                self.subtitle_generator = create_subtitle_generator(self.config)
            subtitle_path = os.path.join(self.temp_dir, "subtitles.srt")
            self.subtitle_generator.generate_srt(translated_segments, subtitle_path)
            self._register_temp_file(subtitle_path)
            logger.info(f"자막 생성 완료: {subtitle_path}")
            
            # 7. 영어 비디오 음소거
            logger.info("\n[7/8] 영어 비디오 음소거 중...")
            if self.video_renderer is None:
                self.video_renderer = create_video_renderer(self.config)
            muted_video_path = os.path.join(self.temp_dir, "muted_video.mp4")
            self.video_renderer.mute_original_audio(english_video_path, muted_video_path)
            self._register_temp_file(muted_video_path)
            
            # 8. 최종 비디오 렌더링
            logger.info("\n[8/8] 최종 비디오 렌더링 중...")
            final_video_path = self.video_renderer.render(
                muted_video_path,
                narration_audio_path,
                subtitle_path,
                output_path
            )
            
            logger.info("=" * 60)
            logger.info("파이프라인 완료!")
            logger.info(f"출력 파일: {final_video_path}")
            logger.info("=" * 60)
            
            return final_video_path
            
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류 발생: {e}", exc_info=True)
            raise
        finally:
            # 임시 파일 정리
            if self.auto_cleanup:
                self.cleanup()


def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

