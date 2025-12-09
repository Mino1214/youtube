"""Whisper STT 모듈 - 영어 오디오를 텍스트로 변환"""

import logging
import torch
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class WhisperSTT:
    """Whisper를 사용한 Speech-to-Text 처리"""
    
    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str = "cuda", 
                 language: str = "en", word_timestamps: bool = True,
                 auto_cpu_fallback: bool = True):
        """
        Args:
            model_name: Whisper 모델 이름 또는 HuggingFace 모델 ID
                       (예: "large-v3", "openai/whisper-large-v3")
            device: "cuda" 또는 "cpu"
            language: 오디오 언어 코드 (기본값: "en")
            word_timestamps: 단어 단위 타임스탬프 사용 여부
            auto_cpu_fallback: CUDA 호환성 문제 시 자동으로 CPU 사용
        """
        # HuggingFace 모델 ID를 openai-whisper 모델명으로 변환
        if "/" in model_name:
            # "openai/whisper-large-v3" -> "large-v3"
            # "openai/whisper-medium" -> "medium"
            model_parts = model_name.split("/")
            if len(model_parts) >= 2:
                model_id = model_parts[-1]  # 마지막 부분
                if model_id.startswith("whisper-"):
                    self.model_name = model_id.replace("whisper-", "")
                else:
                    self.model_name = model_id
            else:
                self.model_name = model_name
        else:
            self.model_name = model_name
        
        self.requested_device = device
        self.language = language
        self.word_timestamps = word_timestamps
        self.auto_cpu_fallback = auto_cpu_fallback
        self.model = None
        self.use_transformers = False  # 항상 openai-whisper 라이브러리 사용
        
        # CUDA 호환성 체크
        if device == "cuda" and torch.cuda.is_available():
            try:
                # 간단한 CUDA 테스트
                test_tensor = torch.randn(10, 10).cuda()
                _ = torch.matmul(test_tensor, test_tensor)
                self.device = "cuda"
                logger.info(f"✅ CUDA 사용 가능 (Compute Capability: {torch.cuda.get_device_capability(0)})")
            except Exception as e:
                if auto_cpu_fallback:
                    logger.warning(f"CUDA 호환성 문제 감지: {e}")
                    logger.warning("자동으로 CPU 모드로 전환합니다.")
                    self.device = "cpu"
                else:
                    raise RuntimeError(f"CUDA 사용 불가: {e}")
        else:
            self.device = "cpu"
        
        logger.info(f"Whisper STT 초기화: 모델={model_name}, 디바이스={self.device}")
    
    def load_model(self):
        """Whisper 모델 로드 (openai-whisper 라이브러리 사용)"""
        if self.model is None:
            logger.info(f"Whisper 모델 로딩 중: {self.model_name} (openai-whisper 라이브러리 사용)")
            
            # openai-whisper 라이브러리 사용
            import whisper
            
            # 모델 로드 (CUDA 12.8 지원 시 GPU 사용 가능)
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    logger.info(f"GPU에서 모델 로드 시도 중...")
                    self.model = whisper.load_model(self.model_name, device="cuda")
                    logger.info("✅ Whisper 모델 CUDA 로드 완료")
                except Exception as e:
                    error_str = str(e).lower()
                    if "no kernel image" in error_str or "cuda" in error_str:
                        logger.warning(f"CUDA 오류 발생: {e}")
                        logger.warning("CPU 모드로 자동 전환합니다...")
                        self.device = "cpu"
                        self.model = whisper.load_model(self.model_name, device="cpu")
                        logger.info("Whisper 모델 CPU 로드 완료")
                    else:
                        raise
            else:
                logger.info(f"CPU에서 모델 로드 중...")
                self.model = whisper.load_model(self.model_name, device="cpu")
                logger.info("Whisper 모델 CPU 로드 완료")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        오디오 파일을 텍스트로 변환
        
        Args:
            audio_path: 오디오 파일 경로
            
        Returns:
            {
                "text": 전체 텍스트,
                "segments": [
                    {
                        "id": 세그먼트 ID,
                        "start": 시작 시간 (초),
                        "end": 종료 시간 (초),
                        "text": 세그먼트 텍스트,
                        "words": [  # word_timestamps가 True인 경우
                            {
                                "word": 단어,
                                "start": 시작 시간,
                                "end": 종료 시간
                            }
                        ]
                    }
                ],
                "language": 감지된 언어
            }
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"오디오 변환 시작: {audio_path}")
        
        # openai-whisper 라이브러리 사용
        # Whisper 옵션 설정
        options = {
            "language": self.language,
            "word_timestamps": self.word_timestamps,
            "verbose": False,
        }
        
        # transcribe 실행 (CUDA 12.8 지원 시 GPU 사용 가능)
        try:
            result = self.model.transcribe(audio_path, **options)
            
            # CUDA 오류 발생 시 CPU로 재시도
        except Exception as e:
            error_str = str(e).lower()
            if "no kernel image" in error_str or "cuda" in error_str:
                logger.error(f"CUDA 오류 발생: {e}")
                logger.warning("CPU 모드로 자동 전환 후 재시도합니다...")
                self.device = "cpu"
                # 모델을 CPU로 이동
                self.model = self.model.cpu()
                # Whisper의 device 속성도 CPU로 설정
                if hasattr(self.model, 'device'):
                    self.model.device = torch.device('cpu')
                # 다시 시도
                result = self.model.transcribe(audio_path, **options)
            else:
                raise
        
        # openai-whisper 라이브러리 결과 처리
        # result가 None이거나 유효하지 않은 경우 체크
        if result is None:
            raise ValueError("Whisper transcribe가 None을 반환했습니다. 오디오 파일을 확인해주세요.")
        
        if not isinstance(result, dict):
            raise ValueError(f"Whisper transcribe가 예상하지 못한 형식을 반환했습니다: {type(result)}")
        
        # 결과 포맷팅
        formatted_result = {
            "text": result.get("text", "").strip() if result.get("text") else "",
            "segments": [],
            "language": result.get("language", self.language)
        }
        
        for i, segment in enumerate(result.get("segments", [])):
            if not isinstance(segment, dict):
                logger.warning(f"세그먼트 {i}가 유효하지 않은 형식입니다: {type(segment)}")
                continue
                
            segment_data = {
                "id": i,
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", "").strip()
            }
            
            if self.word_timestamps and "words" in segment:
                segment_data["words"] = [
                    {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0)
                    }
                    for word in segment["words"]
                    if isinstance(word, dict)
                ]
            
            formatted_result["segments"].append(segment_data)
        
        logger.info(f"변환 완료: {len(formatted_result['segments'])}개 세그먼트")
        return formatted_result
    
    def get_segments_with_timestamps(self, audio_path: str) -> List[Dict[str, Any]]:
        """타임스탬프가 포함된 세그먼트 리스트 반환"""
        result = self.transcribe(audio_path)
        return result["segments"]
    
    def get_full_text(self, audio_path: str) -> str:
        """전체 텍스트만 반환"""
        result = self.transcribe(audio_path)
        return result["text"]


def create_whisper_stt(config: dict) -> WhisperSTT:
    """설정에서 WhisperSTT 인스턴스 생성"""
    whisper_config = config.get("whisper", {})
    return WhisperSTT(
        model_name=whisper_config.get("model", "openai/whisper-large-v3"),
        device=whisper_config.get("device", "cuda"),
        language=whisper_config.get("language", "en"),
        word_timestamps=whisper_config.get("word_timestamps", True),
        auto_cpu_fallback=whisper_config.get("auto_cpu_fallback", True)
    )

