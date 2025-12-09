"""한국어 TTS 모듈 - Piper 및 StyleTTS2 지원"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import platform

logger = logging.getLogger(__name__)


class KoreanTTS:
    """한국어 TTS 생성기"""
    
    def __init__(self, model_type: str = "piper", config: dict = None):
        """
        Args:
            model_type: "piper", "styletts2", 또는 "vibevoice"
            config: TTS 설정 딕셔너리
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        
        logger.info(f"한국어 TTS 초기화: 모델={model_type}")
    
    def load_model(self):
        """TTS 모델 로드"""
        if self.model_type == "piper":
            self._load_piper()
        elif self.model_type == "styletts2":
            self._load_styletts2()
        elif self.model_type == "vibevoice":
            self._load_vibevoice()
        else:
            raise ValueError(f"지원하지 않는 TTS 모델: {self.model_type}")
    
    def _load_piper(self):
        """Piper 모델 로드"""
        try:
            try:
                from piper import PiperVoice
                piper_config = self.config.get("piper", {})
                voice_name = piper_config.get("voice", "neurlang/piper-onnx-kss-korean")
                
                logger.info(f"Piper 음성 로드 중: {voice_name}")
                
                # HuggingFace 모델 ID인지 확인
                is_hf_model = "/" in voice_name and voice_name.count("/") == 1
                
                model_path = None
                config_path = None
                
                if is_hf_model:
                    # HuggingFace 모델 다운로드
                    model_path, config_path = self._download_piper_from_hf(voice_name)
                else:
                    # 기존 방식: 로컬 경로에서 찾기
                    possible_paths = [
                        # 프로젝트 models/tts 디렉토리
                        Path(__file__).parent.parent / "models" / "tts" / f"{voice_name}.onnx",
                        # Piper 기본 위치
                        Path.home() / ".local" / "share" / "piper" / "voices" / voice_name.replace("-", "/") / "model.onnx",
                        # Piper 기본 위치 (다른 형식)
                        Path.home() / ".local" / "share" / "piper" / "voices" / voice_name / "model.onnx",
                    ]
                    
                    # 모델 파일 찾기
                    for path in possible_paths:
                        if path.exists():
                            model_path = path
                            # config 파일도 같은 디렉토리에 있어야 함
                            config_path = path.parent / f"{path.stem}.json"
                            if not config_path.exists():
                                config_path = path.parent / "model.onnx.json"
                            if config_path.exists():
                                break
                            config_path = None
                
                if model_path and config_path and config_path.exists():
                    logger.info(f"Piper 모델 발견: {model_path}")
                    self.model = PiperVoice.load(str(model_path), config_path=str(config_path))
                    logger.info("Piper 모델 로드 완료")
                else:
                    # 모델이 없으면 안내 메시지
                    logger.warning(f"Piper 모델을 찾을 수 없습니다: {voice_name}")
                    if is_hf_model:
                        logger.info("HuggingFace에서 모델 다운로드를 시도합니다...")
                        model_path, config_path = self._download_piper_from_hf(voice_name, force=True)
                        if model_path and config_path:
                            self.model = PiperVoice.load(str(model_path), config_path=str(config_path))
                            logger.info("Piper 모델 로드 완료")
                        else:
                            self.model = None
                    else:
                        logger.info("다음 명령어로 모델을 다운로드하세요:")
                        logger.info(f"  python download_piper_korean.py {voice_name}")
                        logger.info("또는 https://huggingface.co/rhasspy/piper-voices 에서 수동 다운로드")
                        self.model = None
                    
            except ImportError:
                # piper Python 패키지가 없으면 명령줄 도구 사용
                logger.info("piper Python 패키지가 없습니다. 명령줄 도구를 사용합니다.")
                self.model = None
                
        except Exception as e:
            logger.error(f"Piper 모델 로드 실패: {e}")
            raise
    
    def _download_piper_from_hf(self, model_id: str, force: bool = False) -> tuple:
        """
        HuggingFace에서 Piper 모델 다운로드
        
        Args:
            model_id: HuggingFace 모델 ID (예: "neurlang/piper-onnx-kss-korean")
            force: 강제 다운로드 (이미 있어도 다시 다운로드)
            
        Returns:
            (model_path, config_path) 튜플 또는 (None, None)
        """
        try:
            from huggingface_hub import hf_hub_download
            
            # 저장 디렉토리
            save_dir = Path(__file__).parent.parent / "models" / "tts" / model_id.replace("/", "_")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 파일명 추출
            # "neurlang/piper-onnx-kss-korean" -> "piper-kss-korean.onnx"
            repo_id = model_id
            if "/" in model_id:
                repo_id_parts = model_id.split("/")
                model_name = repo_id_parts[-1]  # "piper-onnx-kss-korean"
                
                # 특정 모델에 대한 파일명 매핑
                if model_name == "piper-onnx-kss-korean":
                    # neurlang/piper-onnx-kss-korean의 정확한 파일명
                    file_prefix = "piper-kss-korean"
                else:
                    # 일반적인 경우: "piper-onnx-xxx" -> "piper-xxx"
                    file_prefix = model_name.replace("piper-onnx-", "piper-")
            else:
                file_prefix = model_id
            
            model_file = f"{file_prefix}.onnx"
            config_file = f"{file_prefix}.onnx.json"
            
            model_path = save_dir / model_file
            config_path = save_dir / config_file
            
            # 이미 있으면 스킵
            if not force and model_path.exists() and config_path.exists():
                logger.info(f"Piper 모델이 이미 다운로드되어 있습니다: {model_path}")
                return str(model_path), str(config_path)
            
            logger.info(f"HuggingFace에서 Piper 모델 다운로드 중: {model_id}")
            
            # 모델 파일 다운로드
            try:
                downloaded_model = hf_hub_download(
                    repo_id=model_id,
                    filename=model_file,
                    local_dir=str(save_dir),
                    local_dir_use_symlinks=False
                )
                logger.info(f"모델 파일 다운로드 완료: {downloaded_model}")
            except Exception as e:
                logger.warning(f"모델 파일 다운로드 실패: {e}")
                logger.info("대체 파일명으로 시도 중...")
                # 대체 파일명 시도
                alt_model_file = "model.onnx"
                try:
                    downloaded_model = hf_hub_download(
                        repo_id=model_id,
                        filename=alt_model_file,
                        local_dir=str(save_dir),
                        local_dir_use_symlinks=False
                    )
                    model_path = Path(downloaded_model)
                    logger.info(f"모델 파일 다운로드 완료 (대체): {downloaded_model}")
                except:
                    return None, None
            
            # Config 파일 다운로드
            try:
                downloaded_config = hf_hub_download(
                    repo_id=model_id,
                    filename=config_file,
                    local_dir=str(save_dir),
                    local_dir_use_symlinks=False
                )
                logger.info(f"Config 파일 다운로드 완료: {downloaded_config}")
                config_path = Path(downloaded_config)
            except Exception as e:
                logger.warning(f"Config 파일 다운로드 실패: {e}")
                logger.info("대체 파일명으로 시도 중...")
                # 대체 파일명 시도
                alt_config_file = "model.onnx.json"
                try:
                    downloaded_config = hf_hub_download(
                        repo_id=model_id,
                        filename=alt_config_file,
                        local_dir=str(save_dir),
                        local_dir_use_symlinks=False
                    )
                    config_path = Path(downloaded_config)
                    logger.info(f"Config 파일 다운로드 완료 (대체): {downloaded_config}")
                except:
                    # config 파일이 없으면 모델 파일과 같은 이름으로 생성 시도
                    config_path = model_path.with_suffix(".onnx.json")
                    if not config_path.exists():
                        logger.warning("Config 파일을 찾을 수 없습니다.")
                        return str(model_path), None
            
            if model_path.exists() and config_path.exists():
                logger.info(f"✅ Piper 모델 다운로드 완료: {model_path}")
                return str(model_path), str(config_path)
            else:
                logger.error("모델 또는 config 파일을 찾을 수 없습니다.")
                return None, None
                
        except ImportError:
            logger.error("huggingface_hub 패키지가 필요합니다: pip install huggingface-hub")
            return None, None
        except Exception as e:
            logger.error(f"Piper 모델 다운로드 실패: {e}")
            return None, None
    
    def _load_styletts2(self):
        """StyleTTS2 모델 로드"""
        try:
            # StyleTTS2는 복잡한 초기화가 필요하므로 별도 구현
            logger.warning("StyleTTS2는 아직 완전히 구현되지 않았습니다. Piper를 사용하세요.")
            self.model_type = "piper"
            self._load_piper()
            
        except Exception as e:
            logger.error(f"StyleTTS2 모델 로드 실패: {e}")
            raise
    
    def _load_vibevoice(self):
        """VibeVoice-7B 모델 로드"""
        try:
            import torch
            
            vibevoice_config = self.config.get("vibevoice", {})
            model_id = vibevoice_config.get("model_id", "microsoft/VibeVoice-7B-hf")
            
            # GPU 사용 설정
            use_gpu_config = vibevoice_config.get("use_gpu", True)
            if use_gpu_config:
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    logger.warning("⚠️  VibeVoice GPU 사용 요청했지만 CUDA를 사용할 수 없습니다. CPU 모드로 진행합니다.")
                    device = "cpu"
            else:
                device = "cpu"
            
            logger.info(f"VibeVoice 모델 로드 중: {model_id}")
            logger.info(f"디바이스: {device}, GPU={device == 'cuda'}")
            if device == "cuda":
                logger.info(f"GPU 정보: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
            
            # VibeVoiceForConditionalGeneration 사용 시도
            try:
                from transformers import AutoProcessor, VibeVoiceForConditionalGeneration
                
                # 프로세서 로드 시도
                try:
                    self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                    logger.info("AutoProcessor 로드 성공")
                except Exception as proc_error:
                    logger.warning(f"AutoProcessor 로드 실패: {proc_error}")
                    logger.info("Processor 없이 모델만 로드합니다.")
                    self.processor = None
                
                # 모델 로드
                self.model = VibeVoiceForConditionalGeneration.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )
                
            except (ImportError, AttributeError):
                # VibeVoiceForConditionalGeneration이 없는 경우 AutoModel 사용
                logger.warning("VibeVoiceForConditionalGeneration을 찾을 수 없습니다. AutoModel을 사용합니다.")
                from transformers import AutoProcessor, AutoModel
                
                try:
                    self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                except Exception as proc_error:
                    logger.warning(f"AutoProcessor 로드 실패: {proc_error}")
                    logger.info("Processor 없이 모델만 로드합니다.")
                    self.processor = None
                
                self.model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None
                )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.model.eval()
            self.device = device
            
            logger.info("VibeVoice-7B 모델 로드 완료")
            
        except ImportError as e:
            logger.error(f"VibeVoice를 사용하려면 transformers가 필요합니다: {e}")
            logger.error("또는 특별한 transformers 버전이 필요할 수 있습니다:")
            logger.error("  pip install git+https://github.com/pengzhiliang/transformers.git")
            raise
        except Exception as e:
            logger.error(f"VibeVoice 모델 로드 실패: {e}")
            logger.warning("VibeVoice 모델이 제대로 설치되지 않았을 수 있습니다.")
            logger.warning("Piper 모델로 자동 대체를 시도합니다...")
            
            # Piper로 자동 fallback
            try:
                self.model_type = "piper"
                self.model = None
                self.processor = None
                self._load_piper()
                logger.info("✅ Piper 모델로 성공적으로 대체되었습니다.")
            except Exception as fallback_error:
                logger.error(f"Piper 모델 로드도 실패: {fallback_error}")
                logger.error("해결 방법:")
                logger.error("1. VibeVoice 사용 시: pip install git+https://github.com/pengzhiliang/transformers.git")
                logger.error("2. 또는 config.yaml에서 tts.model을 'piper'로 변경")
                raise RuntimeError(f"VibeVoice 로드 실패: {e}, Piper 대체도 실패: {fallback_error}")
    
    def synthesize(self, text: str, output_path: Optional[str] = None, 
                   speed: Optional[float] = None) -> str:
        """
        텍스트를 오디오로 변환
        
        Args:
            text: 한국어 텍스트
            output_path: 출력 오디오 파일 경로
            speed: 재생 속도 (1.0이 기본값)
            
        Returns:
            생성된 오디오 파일 경로
        """
        if self.model is None:
            self.load_model()
        
        if output_path is None:
            output_path = tempfile.mktemp(suffix='.wav')
        
        if speed is None:
            piper_config = self.config.get("piper", {})
            speed = piper_config.get("speed", 1.0)
        
        logger.info(f"TTS 생성 중: {len(text)}자")
        
        try:
            if self.model_type == "piper":
                return self._synthesize_piper(text, output_path, speed)
            elif self.model_type == "styletts2":
                return self._synthesize_styletts2(text, output_path, speed)
            elif self.model_type == "vibevoice":
                return self._synthesize_vibevoice(text, output_path, speed)
        except Exception as e:
            logger.error(f"TTS 생성 실패: {e}")
            raise
    
    def _synthesize_piper(self, text: str, output_path: str, speed: float) -> str:
        """Piper를 사용한 TTS 생성"""
        piper_config = self.config.get("piper", {})
        voice_name = piper_config.get("voice", "ko_KR-hyeri-medium")
        
        # 모델이 없으면 다시 로드 시도
        if self.model is None:
            logger.info("모델이 없습니다. 다시 로드 시도...")
            self._load_piper()
        
        if self.model is not None:
            # Python API 사용
            noise_scale = piper_config.get("noise_scale", 0.667)
            length_penalty = piper_config.get("length_penalty", 1.0)
            
            try:
                with open(output_path, "wb") as f:
                    self.model.synthesize(
                        text,
                        f,
                        length_penalty=length_penalty,
                        noise_scale=noise_scale,
                        noise_w=0.8,
                        speed=speed
                    )
                logger.info(f"TTS 생성 완료: {output_path}")
                return output_path
            except Exception as e:
                logger.error(f"Piper TTS 생성 실패: {e}")
                raise
        else:
            # 모델을 찾을 수 없으면 에러
            error_msg = (
                f"Piper TTS 모델을 찾을 수 없습니다.\n"
                f"음성: {voice_name}\n"
                f"해결 방법:\n"
                f"1. 다음 명령어로 모델 다운로드:\n"
                f"   python -m piper.download --voice {voice_name}\n"
                f"2. 또는 https://huggingface.co/rhasspy/piper-voices 에서 수동 다운로드"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _synthesize_styletts2(self, text: str, output_path: str, speed: float) -> str:
        """StyleTTS2를 사용한 TTS 생성 (미구현)"""
        raise NotImplementedError("StyleTTS2는 아직 구현되지 않았습니다")
    
    def _synthesize_vibevoice(self, text: str, output_path: str, speed: float) -> str:
        """VibeVoice-7B를 사용한 TTS 생성"""
        import torch
        import soundfile as sf
        import numpy as np
        
        vibevoice_config = self.config.get("vibevoice", {})
        
        try:
            # 텍스트 처리
            if self.processor is not None:
                # Processor 사용
                inputs = self.processor(text=text, return_tensors="pt")
                # 디바이스로 이동
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
            else:
                # Processor 없이 직접 텍스트 사용 (모델에 따라 다를 수 있음)
                logger.warning("Processor가 없습니다. 직접 텍스트를 사용합니다.")
                inputs = {"text": text}
            
            # TTS 생성
            with torch.no_grad():
                if isinstance(inputs, dict):
                    outputs = self.model.generate(**inputs)
                else:
                    outputs = self.model.generate(inputs)
            
            # 오디오 추출
            audio = None
            
            # VibeVoiceForConditionalGeneration의 경우
            if hasattr(outputs, 'speech_outputs'):
                audio = outputs.speech_outputs[0].cpu().numpy()
            elif hasattr(outputs, 'audio'):
                audio = outputs.audio.cpu().numpy()
                if len(audio.shape) > 1:
                    audio = audio[0] if audio.shape[0] == 1 else audio
            elif isinstance(outputs, torch.Tensor):
                audio = outputs.cpu().numpy()
            elif isinstance(outputs, (tuple, list)):
                # 첫 번째 요소가 오디오일 가능성
                if len(outputs) > 0:
                    first_item = outputs[0]
                    if isinstance(first_item, torch.Tensor):
                        audio = first_item.cpu().numpy()
                    elif hasattr(first_item, 'speech_outputs'):
                        audio = first_item.speech_outputs[0].cpu().numpy()
            elif isinstance(outputs, dict):
                # 딕셔너리에서 오디오 찾기
                audio = outputs.get('speech_outputs', outputs.get('audio', outputs.get('values', None)))
                if audio is not None:
                    if isinstance(audio, (list, tuple)) and len(audio) > 0:
                        audio = audio[0]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
            
            if audio is None:
                raise ValueError(f"오디오 데이터를 추출할 수 없습니다. 출력 형식: {type(outputs)}")
            
            # 1D 배열로 변환
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # 속도 조정
            if speed != 1.0:
                try:
                    import scipy.signal
                    num_samples = int(len(audio) / speed)
                    audio = scipy.signal.resample(audio, num_samples)
                except ImportError:
                    logger.warning("scipy가 없어 속도 조정을 건너뜁니다.")
            
            # 샘플레이트 (VibeVoice는 보통 24kHz)
            sample_rate = 24000
            
            # 오디오 저장
            sf.write(output_path, audio, sample_rate)
            
            logger.info(f"VibeVoice TTS 생성 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"VibeVoice TTS 생성 실패: {e}")
            logger.error("VibeVoice 모델이 제대로 설치되지 않았거나 API가 변경되었을 수 있습니다.")
            logger.info("해결 방법:")
            logger.info("1. 특별한 transformers 버전 설치:")
            logger.info("   pip install git+https://github.com/pengzhiliang/transformers.git")
            logger.info("2. 또는 config.yaml에서 tts.model을 'piper'로 변경하여 Piper 사용")
            raise
    
    def synthesize_segments(self, segments: List[Dict[str, Any]], 
                           output_dir: Optional[str] = None) -> List[str]:
        """
        세그먼트 리스트를 오디오로 변환
        
        Args:
            segments: 번역된 세그먼트 리스트
            output_dir: 출력 디렉토리
            
        Returns:
            생성된 오디오 파일 경로 리스트
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_files = []
        
        logger.info(f"{len(segments)}개 세그먼트 TTS 생성 시작")
        
        for i, segment in enumerate(segments):
            korean_text = segment.get("text_ko", segment.get("text", ""))
            if not korean_text:
                continue
            
            output_path = str(output_dir / f"segment_{i:04d}.wav")
            
            try:
                audio_path = self.synthesize(korean_text, output_path)
                audio_files.append(audio_path)
            except Exception as e:
                logger.error(f"세그먼트 {i} TTS 생성 실패: {e}")
                # 빈 오디오 파일 생성 (타임코드 유지)
                self._create_silent_audio(output_path, segment["end"] - segment["start"])
                audio_files.append(output_path)
        
        logger.info(f"TTS 생성 완료: {len(audio_files)}개 파일")
        return audio_files
    
    def _create_silent_audio(self, output_path: str, duration: float):
        """무음 오디오 파일 생성 (에러 시 fallback)"""
        try:
            import subprocess
            cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=mono:sample_rate=22050',
                '-t', str(duration),
                '-y',
                output_path
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        except Exception as e:
            logger.warning(f"무음 오디오 생성 실패: {e}")
    
    def merge_audio_segments(self, audio_files: List[str], 
                            segments: List[Dict[str, Any]], 
                            output_path: str) -> str:
        """
        세그먼트 오디오 파일들을 타임코드에 맞춰 병합
        
        Args:
            audio_files: 오디오 파일 경로 리스트
            segments: 세그먼트 정보 (타임코드 포함)
            output_path: 최종 출력 경로
            
        Returns:
            병합된 오디오 파일 경로
        """
        logger.info("오디오 세그먼트 병합 중...")
        
        try:
            import subprocess
            import tempfile
            
            # FFmpeg 필터 복잡도로 인해 간단한 방법 사용
            # 모든 오디오를 순차적으로 연결
            concat_file = tempfile.mktemp(suffix='.txt')
            
            with open(concat_file, 'w', encoding='utf-8') as f:
                for audio_file in audio_files:
                    f.write(f"file '{os.path.abspath(audio_file)}'\n")
            
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                '-y',
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # 타임코드에 맞춰 조정 (필요시)
            # 실제로는 각 세그먼트 사이에 무음을 삽입해야 할 수 있음
            
            logger.info(f"오디오 병합 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"오디오 병합 실패: {e}")
            raise


def create_korean_tts(config: dict) -> KoreanTTS:
    """설정에서 KoreanTTS 인스턴스 생성"""
    tts_config = config.get("tts", {})
    return KoreanTTS(
        model_type=tts_config.get("model", "piper"),
        config=tts_config
    )

