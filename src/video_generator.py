"""외국어 비디오 생성 모듈 - 텍스트에서 비디오 생성 (VEO3 수준)"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

logger = logging.getLogger(__name__)


class VideoGenerator:
    """텍스트에서 외국어 비디오 생성기 (VEO3 수준)"""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: 비디오 생성 설정
        """
        self.config = config or {}
        video_gen_config = self.config.get("video_generation", {})
        self.width = video_gen_config.get("width", 1920)
        self.height = video_gen_config.get("height", 1080)
        self.fps = video_gen_config.get("fps", 30)
        self.background_color = video_gen_config.get("background_color", "#000000")
        self.use_image_generation = video_gen_config.get("use_image_generation", False)
        self.max_duration = video_gen_config.get("max_duration")  # 채널별 최대 길이 제한
        
        # 비디오 생성 모델 설정
        self.video_model = video_gen_config.get("model", "svd")  # "svd", "animatediff", "simple"
        self.video_model_path = video_gen_config.get("model_path", None)
        
        # GPU 사용 설정
        config_use_gpu = video_gen_config.get("use_gpu", True)
        force_gpu = video_gen_config.get("force_gpu", False)  # 강제 GPU 모드
        auto_cpu_fallback = video_gen_config.get("auto_cpu_fallback", True)  # 자동 CPU fallback
        
        # CUDA 사용 가능 여부 확인
        cuda_available = torch.cuda.is_available()
        
        # GPU 초기화 및 검증
        self.use_gpu = False
        self.device = "cpu"
        self.gpu_error = None
        
        if config_use_gpu or force_gpu:
            if not cuda_available:
                if not auto_cpu_fallback:
                    error_msg = "CUDA를 사용할 수 없습니다."
                    logger.error(f"❌ {error_msg}")
                    logger.error("해결 방법:")
                    logger.error("  1. NVIDIA 드라이버 설치 확인: nvidia-smi")
                    logger.error("  2. PyTorch GPU 버전 설치:")
                    logger.error("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
                    logger.error("  3. CUDA Toolkit 설치 확인")
                    raise RuntimeError(error_msg)
                else:
                    logger.warning("⚠️  CUDA를 사용할 수 없습니다. CPU 모드로 전환합니다.")
                    self.use_gpu = False
                    self.device = "cpu"
            else:
                try:
                    # GPU 초기화 테스트
                    logger.info("GPU 초기화 테스트 중...")
                    test_tensor = torch.zeros(1).cuda()
                    test_result = test_tensor * 2
                    del test_tensor, test_result
                    torch.cuda.empty_cache()
                    
                    # GPU 정보 확인
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_capability = torch.cuda.get_device_capability(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    
                    self.use_gpu = True
                    self.device = "cuda"
                    logger.info(f"✅ CUDA GPU 초기화 성공")
                    logger.info(f"   GPU: {gpu_name}")
                    logger.info(f"   Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
                    logger.info(f"   메모리: {gpu_memory:.2f} GB")
                    logger.info(f"   CUDA 버전: {torch.version.cuda}")
                    
                    # Compute capability 경고
                    if gpu_capability[0] < 7:
                        logger.warning(f"⚠️  GPU Compute Capability가 낮습니다 ({gpu_capability[0]}.{gpu_capability[1]})")
                        logger.warning("   일부 모델이 작동하지 않을 수 있습니다.")
                    
                except RuntimeError as e:
                    error_str = str(e).lower()
                    if "no kernel image" in error_str or "kernel image" in error_str:
                        self.gpu_error = "GPU compute capability 호환성 문제"
                        logger.error("=" * 60)
                        logger.error("❌ GPU Compute Capability 호환성 오류")
                        logger.error("=" * 60)
                        logger.error(f"오류: {e}")
                        logger.error("")
                        logger.error("원인: PyTorch가 현재 GPU의 compute capability를 지원하지 않습니다.")
                        logger.error("")
                        
                        if auto_cpu_fallback:
                            logger.warning("⚠️  CPU 모드로 자동 전환합니다.")
                            logger.warning("   처리 속도가 느려질 수 있습니다.")
                            self.use_gpu = False
                            self.device = "cpu"
                            self.gpu_error = None
                        else:
                            logger.error("해결 방법:")
                            logger.error("  1. PyTorch Nightly 버전 시도:")
                            logger.error("     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                            logger.error("")
                            logger.error("  2. 또는 소스에서 빌드:")
                            logger.error("     https://pytorch.org/get-started/locally/")
                            logger.error("")
                            logger.error("  3. 또는 config.yaml에서 auto_cpu_fallback: true로 설정")
                            logger.error("=" * 60)
                            raise RuntimeError(f"GPU compute capability 호환성 오류: {e}")
                    else:
                        self.gpu_error = str(e)
                        if auto_cpu_fallback:
                            logger.warning(f"⚠️  GPU 초기화 실패: {e}")
                            logger.warning("⚠️  CPU 모드로 자동 전환합니다.")
                            self.use_gpu = False
                            self.device = "cpu"
                            self.gpu_error = None
                        else:
                            logger.error(f"❌ GPU 초기화 실패: {e}")
                            raise RuntimeError(f"GPU 초기화 실패: {e}")
                except Exception as e:
                    self.gpu_error = str(e)
                    if auto_cpu_fallback:
                        logger.warning(f"⚠️  GPU 초기화 중 예상치 못한 오류: {e}")
                        logger.warning("⚠️  CPU 모드로 자동 전환합니다.")
                        self.use_gpu = False
                        self.device = "cpu"
                        self.gpu_error = None
                    else:
                        logger.error(f"❌ GPU 초기화 중 예상치 못한 오류: {e}")
                        raise RuntimeError(f"GPU 초기화 실패: {e}")
        else:
            logger.info("CPU 모드로 실행합니다.")
            self.use_gpu = False
            self.device = "cpu"
        
        # 모델 인스턴스 (지연 로딩)
        self.svd_model = None
        self.animatediff_model = None
        self.stable_diffusion_model = None
        
        logger.info(f"비디오 생성기 초기화: 모델={self.video_model}, 디바이스={self.device}, GPU={self.use_gpu}")
    
    def generate_from_text(self, text: str, audio_path: str, 
                         output_path: str, duration: Optional[float] = None) -> str:
        """
        텍스트와 오디오로부터 비디오 생성
        
        Args:
            text: 표시할 텍스트
            audio_path: 오디오 파일 경로
            output_path: 출력 비디오 경로
            duration: 비디오 길이 (초, None이면 오디오 길이 사용)
            
        Returns:
            생성된 비디오 파일 경로
        """
        logger.info(f"비디오 생성 중: {output_path}")
        
        # 오디오 길이 확인
        if duration is None:
            duration = self._get_audio_duration(audio_path)
        
        # 채널별 최대 길이 제한 적용
        if self.max_duration and duration > self.max_duration:
            logger.warning(f"비디오 길이({duration:.1f}초)가 최대 길이({self.max_duration}초)를 초과합니다. {self.max_duration}초로 제한합니다.")
            duration = self.max_duration
        
        # 비디오 생성 모델에 따라 분기
        if self.video_model == "svd":
            # Stable Video Diffusion: 텍스트 → 이미지 → 비디오
            return self._generate_with_svd(text, audio_path, output_path, duration)
        elif self.video_model == "animatediff":
            # AnimateDiff: 텍스트 → 직접 비디오
            return self._generate_with_animatediff(text, audio_path, output_path, duration)
        elif self.use_image_generation:
            # 기존 이미지 생성 방식
            return self._generate_with_images(text, audio_path, output_path, duration)
        else:
            # 간단한 텍스트 슬라이드 비디오 (fallback)
            return self._generate_simple_slideshow(text, audio_path, output_path, duration)
    
    def _generate_simple_slideshow(self, text: str, audio_path: str, 
                                   output_path: str, duration: float) -> str:
        """간단한 슬라이드쇼 비디오 생성 (텍스트 표시)"""
        logger.info("슬라이드쇼 비디오 생성 중...")
        
        # 텍스트를 여러 줄로 분할
        lines = self._split_text_to_lines(text, max_chars_per_line=50)
        
        # 이미지 생성
        image_path = tempfile.mktemp(suffix='.png')
        self._create_text_image(lines, image_path)
        
        # 이미지를 비디오로 변환
        video_path = self._image_to_video(image_path, audio_path, output_path, duration)
        
        # 임시 이미지 삭제
        if os.path.exists(image_path):
            os.remove(image_path)
        
        return video_path
    
    def _create_text_image(self, lines: List[str], output_path: str):
        """텍스트가 포함된 이미지 생성"""
        # 이미지 생성
        img = Image.new('RGB', (self.width, self.height), color=self.background_color)
        draw = ImageDraw.Draw(img)
        
        # 폰트 설정 (시스템 폰트 사용)
        try:
            # Windows 기본 폰트
            font_size = 60
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # Linux 기본 폰트
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                # 기본 폰트
                font = ImageFont.load_default()
        
        # 텍스트 그리기
        text_color = "#FFFFFF"
        y_offset = self.height // 2 - (len(lines) * font_size) // 2
        
        for i, line in enumerate(lines):
            # 텍스트 중앙 정렬
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (self.width - text_width) // 2
            y = y_offset + i * (font_size + 20)
            
            # 텍스트 그리기 (외곽선 포함)
            draw.text((x, y), line, fill=text_color, font=font)
        
        # 이미지 저장
        img.save(output_path)
        logger.debug(f"텍스트 이미지 생성 완료: {output_path}")
    
    def _split_text_to_lines(self, text: str, max_chars_per_line: int = 50) -> List[str]:
        """텍스트를 여러 줄로 분할"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # 공백 포함
            if current_length + word_length > max_chars_per_line and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    def _image_to_video(self, image_path: str, audio_path: str, 
                       output_path: str, duration: float) -> str:
        """이미지와 오디오를 결합하여 비디오 생성"""
        logger.info(f"이미지+오디오 → 비디오 변환 중...")
        
        # 오디오 파일 존재 및 유효성 확인
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            logger.warning(f"오디오 파일이 없습니다: {audio_path}. 무음 오디오로 재생성합니다.")
            self._create_silent_audio(audio_path, duration)
        elif audio_path_obj.stat().st_size == 0:
            logger.warning(f"오디오 파일이 비어있습니다. 무음 오디오로 재생성합니다.")
            self._create_silent_audio(audio_path, duration)
        else:
            # 오디오 파일 유효성 검사 (FFprobe로)
            try:
                probe_cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    audio_path
                ]
                probe_result = subprocess.run(
                    probe_cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if probe_result.returncode != 0:
                    logger.warning(f"오디오 파일이 손상되었습니다. 무음 오디오로 재생성합니다.")
                    self._create_silent_audio(audio_path, duration)
            except Exception as e:
                logger.warning(f"오디오 파일 검증 실패: {e}. 계속 진행합니다.")
        
        cmd = [
            'ffmpeg',
            '-loop', '1',
            '-i', image_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-t', str(duration),
            '-pix_fmt', 'yuv420p',
            '-shortest',
            '-y',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
            logger.info(f"비디오 생성 완료: {output_path}")
            return output_path
        except subprocess.TimeoutExpired:
            logger.error("비디오 생성 시간 초과")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"비디오 생성 실패:")
            logger.error(f"  명령어: {' '.join(cmd)}")
            logger.error(f"  오류: {e.stderr}")
            if "Invalid data" in e.stderr or "Error opening input" in e.stderr:
                logger.warning("오디오 파일 문제로 보입니다. 무음 오디오로 재시도합니다.")
                self._create_silent_audio(audio_path, duration)
                # 재시도
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
                    logger.info(f"비디오 생성 완료 (재시도): {output_path}")
                    return output_path
                except Exception as e2:
                    logger.error(f"재시도도 실패: {e2}")
                    raise
            raise
        except FileNotFoundError:
            logger.error("FFmpeg를 찾을 수 없습니다.")
            raise
    
    def _generate_with_svd(self, text: str, audio_path: str, 
                          output_path: str, duration: float) -> str:
        """Stable Video Diffusion을 사용한 비디오 생성"""
        logger.info("Stable Video Diffusion으로 비디오 생성 중...")
        
        try:
            # 1. 텍스트에서 이미지 생성 (Stable Diffusion)
            logger.info("1단계: 텍스트에서 이미지 생성 중...")
            image_path = self._generate_image_from_text(text)
            
            # 2. 이미지에서 비디오 생성 (SVD)
            logger.info("2단계: 이미지에서 비디오 생성 중...")
            video_path = self._generate_video_from_image_svd(image_path, duration)
            
            # 3. 오디오와 결합
            logger.info("3단계: 오디오와 비디오 결합 중...")
            final_path = self._combine_video_audio(video_path, audio_path, output_path)
            
            # 임시 파일 정리
            if os.path.exists(image_path):
                os.remove(image_path)
            if video_path != final_path and os.path.exists(video_path):
                os.remove(video_path)
            
            logger.info(f"✅ SVD 비디오 생성 완료: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"SVD 비디오 생성 실패: {e}")
            logger.warning("간단한 슬라이드쇼로 대체합니다.")
            return self._generate_simple_slideshow(text, audio_path, output_path, duration)
    
    def _generate_with_animatediff(self, text: str, audio_path: str,
                                   output_path: str, duration: float) -> str:
        """AnimateDiff를 사용한 텍스트→비디오 직접 생성"""
        logger.info("AnimateDiff로 비디오 생성 중...")
        
        try:
            # AnimateDiff로 직접 비디오 생성
            video_path = self._generate_video_from_text_animatediff(text, duration)
            
            # 오디오와 결합
            final_path = self._combine_video_audio(video_path, audio_path, output_path)
            
            # 임시 파일 정리
            if video_path != final_path and os.path.exists(video_path):
                os.remove(video_path)
            
            logger.info(f"✅ AnimateDiff 비디오 생성 완료: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"AnimateDiff 비디오 생성 실패: {e}")
            logger.warning("간단한 슬라이드쇼로 대체합니다.")
            return self._generate_simple_slideshow(text, audio_path, output_path, duration)
    
    def _generate_image_from_text(self, text: str) -> str:
        """Stable Diffusion으로 텍스트에서 이미지 생성"""
        if self.stable_diffusion_model is None:
            self._load_stable_diffusion()
        
        logger.info(f"이미지 생성 중: {text[:50]}...")
        
        try:
            # 프롬프트 생성
            prompt = text[:500]  # 프롬프트 길이 제한
            
            # 이미지 생성 (SDXL은 더 큰 해상도 지원)
            # SDXL 권장 해상도: 1024x1024 또는 1024x768 (16:9)
            gen_width = max(1024, self.width) if self.width >= 1024 else 1024
            gen_height = max(1024, self.height) if self.height >= 1024 else 1024
            
            # 16:9 비율 유지
            if gen_width / gen_height > 1.5:  # 가로가 더 긴 경우
                gen_height = int(gen_width / 16 * 9)
            elif gen_height / gen_width > 1.5:  # 세로가 더 긴 경우
                gen_width = int(gen_height / 9 * 16)
            
            logger.info(f"이미지 생성 해상도: {gen_width}x{gen_height}")
            
            try:
                result = self.stable_diffusion_model(
                    prompt=prompt,
                    num_inference_steps=30,  # SDXL은 더 많은 스텝 권장
                    guidance_scale=7.5,
                    width=gen_width,
                    height=gen_height
                )
                image = result.images[0]
            except RuntimeError as e:
                error_str = str(e).lower()
                if "cuda" in error_str or "kernel image" in error_str or "no kernel" in error_str:
                    logger.error("=" * 60)
                    logger.error("❌ GPU CUDA 오류 발생 (이미지 생성)")
                    logger.error("=" * 60)
                    logger.error(f"오류: {e}")
                    logger.error("")
                    logger.error("원인: PyTorch가 현재 GPU의 compute capability를 지원하지 않습니다.")
                    logger.error("")
                    logger.error("해결 방법:")
                    logger.error("  1. GPU 호환성 확인:")
                    logger.error("     check_gpu_compatibility.bat")
                    logger.error("")
                    logger.error("  2. PyTorch Nightly 버전 설치:")
                    logger.error("     install_pytorch_nightly.bat")
                    logger.error("")
                    logger.error("  3. GPU Compute Capability 확인:")
                    logger.error("     python -c \"import torch; print('Capability:', torch.cuda.get_device_capability(0))\"")
                    logger.error("=" * 60)
                    raise RuntimeError(f"GPU CUDA 오류 (이미지 생성): {e}")
                else:
                    raise
            
            # 원하는 해상도로 리사이즈
            if gen_width != self.width or gen_height != self.height:
                image = image.resize((self.width, self.height), Image.LANCZOS)
            
            # 이미지 저장
            image_path = tempfile.mktemp(suffix='.png')
            image.save(image_path)
            logger.info(f"이미지 생성 완료: {image_path}")
            return image_path
            
        except RuntimeError as e:
            error_str = str(e).lower()
            if "cuda" in error_str or "kernel image" in error_str or "no kernel" in error_str:
                logger.error("=" * 60)
                logger.error("❌ GPU CUDA 오류 발생 (이미지 생성)")
                logger.error("=" * 60)
                logger.error(f"오류: {e}")
                logger.error("")
                logger.error("원인: PyTorch가 현재 GPU의 compute capability를 지원하지 않습니다.")
                logger.error("")
                logger.error("해결 방법:")
                logger.error("  1. GPU 호환성 확인: check_gpu_compatibility.bat")
                logger.error("  2. PyTorch Nightly 설치: install_pytorch_nightly.bat")
                logger.error("=" * 60)
                raise RuntimeError(f"GPU CUDA 오류 (이미지 생성): {e}")
        except ImportError:
            logger.error("diffusers 패키지가 설치되지 않았습니다.")
            logger.info("설치: pip install diffusers")
            raise
        except Exception as e:
            logger.error(f"이미지 생성 실패: {e}")
            raise
    
    def _load_stable_diffusion(self):
        """Stable Diffusion XL 모델 로드 (고품질)"""
        try:
            # transformers 버전 확인
            try:
                from transformers import CLIPImageProcessor
                logger.debug("✅ CLIPImageProcessor 사용 가능")
            except ImportError:
                logger.error("❌ transformers 버전이 너무 낮습니다!")
                logger.error("다음 명령어로 업그레이드하세요:")
                logger.error("  pip install --upgrade transformers>=4.40.0")
                logger.error("또는 fix_dependencies.py를 실행하세요.")
                raise ImportError(
                    "transformers 버전이 낮습니다. "
                    "CLIPImageProcessor를 사용하려면 transformers>=4.40.0이 필요합니다. "
                    "업그레이드: pip install --upgrade transformers>=4.40.0"
                )
            
            from diffusers import DiffusionPipeline
            
            # Stable Diffusion XL 사용 (더 높은 품질)
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            logger.info(f"Stable Diffusion XL 모델 로드 중: {model_id}")
            logger.info("⚠️  첫 로드는 시간이 오래 걸릴 수 있습니다 (약 7GB 다운로드)")
            
            # GPU 사용 가능 여부 재확인
            if self.use_gpu and not torch.cuda.is_available():
                logger.error("❌ GPU 모드이지만 CUDA를 사용할 수 없습니다.")
                raise RuntimeError("CUDA를 사용할 수 없습니다. GPU를 사용하려면 CUDA를 설치하세요.")
            
            try:
                self.stable_diffusion_model = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if self.use_gpu else None,
                    device_map="auto" if self.use_gpu else None
                )
                
                # 디바이스로 이동 시도
                try:
                    if not self.use_gpu:
                        self.stable_diffusion_model = self.stable_diffusion_model.to(self.device)
                    elif self.use_gpu and not hasattr(self.stable_diffusion_model, 'device_map'):
                        self.stable_diffusion_model = self.stable_diffusion_model.to(self.device)
                except Exception as device_error:
                    error_str = str(device_error).lower()
                    if "cuda" in error_str or "kernel image" in error_str or "no kernel" in error_str:
                        logger.error("=" * 60)
                        logger.error("❌ GPU CUDA 오류 발생 (모델 로드)")
                        logger.error("=" * 60)
                        logger.error(f"오류: {device_error}")
                        logger.error("")
                        logger.error("해결 방법: install_pytorch_nightly.bat 실행")
                        logger.error("=" * 60)
                        raise RuntimeError(f"GPU CUDA 오류 (모델 로드): {device_error}")
                    else:
                        logger.error(f"❌ 모델 이동 실패: {device_error}")
                        raise
                
                # 메모리 최적화
                if self.use_gpu:
                    try:
                        self.stable_diffusion_model.enable_model_cpu_offload()
                        logger.info("CPU 오프로딩 활성화 (메모리 절약)")
                    except:
                        pass
                
                logger.info(f"✅ Stable Diffusion XL 모델 로드 완료 (디바이스: {self.device})")
                
            except RuntimeError as e:
                error_str = str(e).lower()
                if "cuda" in error_str or "gpu" in error_str or "kernel image" in error_str or "no kernel" in error_str:
                    logger.error("=" * 60)
                    logger.error("❌ Stable Diffusion XL GPU 로드 오류")
                    logger.error("=" * 60)
                    logger.error(f"오류: {e}")
                    logger.error("")
                    logger.error("원인: PyTorch가 현재 GPU의 compute capability를 지원하지 않습니다.")
                    logger.error("")
                    logger.error("해결 방법:")
                    logger.error("  1. PyTorch Nightly 버전 설치:")
                    logger.error("     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                    logger.error("")
                    logger.error("  2. GPU 정보 확인:")
                    logger.error("     python -c \"import torch; print('GPU:', torch.cuda.get_device_name(0)); print('Capability:', torch.cuda.get_device_capability(0))\"")
                    logger.error("=" * 60)
                    raise RuntimeError(f"GPU 로드 오류: {e}")
                else:
                    raise
            
        except ImportError as e:
            error_msg = str(e)
            if "CLIPImageProcessor" in error_msg or "transformers" in error_msg.lower():
                logger.error("=" * 60)
                logger.error("의존성 버전 문제 감지!")
                logger.error("=" * 60)
                logger.error("\n해결 방법:")
                logger.error("1. fix_dependencies.py 실행:")
                logger.error("   python fix_dependencies.py")
                logger.error("\n2. 또는 수동으로 업그레이드:")
                logger.error("   pip install --upgrade transformers>=4.40.0 diffusers>=0.27.0")
                logger.error("=" * 60)
            else:
                logger.error("diffusers 패키지가 필요합니다: pip install diffusers")
            raise
        except Exception as e:
            logger.error(f"Stable Diffusion XL 로드 실패: {e}")
            logger.warning("기본 Stable Diffusion v1.5로 대체 시도 중...")
            # Fallback to SD 1.5
            try:
                from diffusers import StableDiffusionPipeline
                model_id = "runwayml/stable-diffusion-v1-5"
                logger.info(f"대체 모델 로드: {model_id}")
                self.stable_diffusion_model = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                )
                self.stable_diffusion_model = self.stable_diffusion_model.to(self.device)
                logger.info("✅ Stable Diffusion v1.5 로드 완료")
            except Exception as e2:
                logger.error(f"대체 모델 로드도 실패: {e2}")
                raise
    
    def _generate_video_from_image_svd(self, image_path: str, duration: float) -> str:
        """Stable Video Diffusion으로 이미지에서 비디오 생성"""
        if self.svd_model is None:
            self._load_svd()
        
        logger.info(f"SVD 비디오 생성 중: {image_path}")
        
        try:
            from diffusers import StableVideoDiffusionPipeline
            from diffusers.utils import load_image
            
            # 이미지 로드 및 리사이즈
            image = load_image(image_path)
            image = image.resize((self.width, self.height))
            
            # 비디오 생성
            result = self.svd_model(
                image,
                decode_chunk_size=8,
                num_frames=min(int(duration * self.fps), 25),  # 최대 25프레임
                num_inference_steps=25,
                motion_bucket_id=127,
            )
            frames = result.frames[0]
            
            # 비디오 저장
            video_path = tempfile.mktemp(suffix='.mp4')
            self._frames_to_video(frames, video_path, self.fps)
            
            logger.info(f"SVD 비디오 생성 완료: {video_path}")
            return video_path
            
        except ImportError:
            logger.error("diffusers 패키지가 설치되지 않았습니다.")
            logger.info("설치: pip install diffusers")
            raise
        except Exception as e:
            logger.error(f"SVD 비디오 생성 실패: {e}")
            raise
    
    def _load_svd(self):
        """Stable Video Diffusion 모델 로드"""
        try:
            from diffusers import StableVideoDiffusionPipeline
            
            model_id = self.video_model_path or "stabilityai/stable-video-diffusion-img2vid"
            logger.info(f"SVD 모델 로드 중: {model_id}")
            
            # GPU 사용 가능 여부 재확인
            if self.use_gpu and not torch.cuda.is_available():
                logger.error("❌ GPU 모드이지만 CUDA를 사용할 수 없습니다.")
                raise RuntimeError("CUDA를 사용할 수 없습니다. GPU를 사용하려면 CUDA를 설치하세요.")
            
            try:
                self.svd_model = StableVideoDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                    variant="fp16" if self.use_gpu else None
                )
                
                # 디바이스로 이동 시도
                try:
                    self.svd_model = self.svd_model.to(self.device)
                except Exception as device_error:
                    error_str = str(device_error).lower()
                    if "cuda" in error_str or "kernel image" in error_str or "no kernel" in error_str:
                        logger.error("=" * 60)
                        logger.error("❌ SVD GPU 모델 이동 오류")
                        logger.error("=" * 60)
                        logger.error(f"오류: {device_error}")
                        logger.error("")
                        logger.error("해결 방법: install_pytorch_nightly.bat 실행")
                        logger.error("=" * 60)
                        raise RuntimeError(f"SVD GPU 모델 이동 오류: {device_error}")
                    else:
                        logger.error(f"❌ 모델 이동 실패: {device_error}")
                        raise
                
                logger.info(f"✅ SVD 모델 로드 완료 (디바이스: {self.device})")
                
            except RuntimeError as e:
                error_str = str(e).lower()
                if "cuda" in error_str or "gpu" in error_str or "kernel image" in error_str or "no kernel" in error_str:
                    logger.error("=" * 60)
                    logger.error("❌ SVD GPU 로드 오류")
                    logger.error("=" * 60)
                    logger.error(f"오류: {e}")
                    logger.error("")
                    logger.error("원인: PyTorch가 현재 GPU의 compute capability를 지원하지 않습니다.")
                    logger.error("")
                    logger.error("해결 방법:")
                    logger.error("  1. PyTorch Nightly 버전 설치:")
                    logger.error("     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                    logger.error("=" * 60)
                    raise RuntimeError(f"SVD GPU 로드 오류: {e}")
                else:
                    raise
            
        except ImportError:
            logger.error("diffusers 패키지가 필요합니다: pip install diffusers")
            raise
        except Exception as e:
            logger.error(f"❌ SVD 로드 실패: {e}")
            raise
    
    def _generate_video_from_text_animatediff(self, text: str, duration: float) -> str:
        """AnimateDiff로 텍스트에서 직접 비디오 생성"""
        if self.animatediff_model is None:
            self._load_animatediff()
        
        logger.info(f"AnimateDiff 비디오 생성 중: {text[:50]}...")
        
        try:
            # AnimateDiff는 텍스트 프롬프트를 직접 받음
            prompt = text[:500]
            num_frames = min(int(duration * self.fps), 16)  # 최대 16프레임
            
            # 비디오 생성
            result = self.animatediff_model(
                prompt=prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                num_frames=num_frames,
                height=self.height,
                width=self.width,
            )
            
            # frames 추출
            if hasattr(result, 'frames'):
                frames = result.frames
            elif hasattr(result, 'images'):
                frames = result.images
            else:
                frames = result
            
            # 비디오 저장
            video_path = tempfile.mktemp(suffix='.mp4')
            self._frames_to_video(frames, video_path, self.fps)
            
            logger.info(f"AnimateDiff 비디오 생성 완료: {video_path}")
            return video_path
            
        except Exception as e:
            logger.error(f"AnimateDiff 비디오 생성 실패: {e}")
            raise
    
    def _load_animatediff(self):
        """AnimateDiff 모델 로드"""
        try:
            # AnimateDiff는 여러 구현 방식이 있음
            # 방법 1: diffusers의 AnimateDiffPipeline 사용
            try:
                from diffusers import AnimateDiffPipeline, DDIMScheduler
                from diffusers.utils import export_to_video
                from transformers import CLIPTextModel, CLIPTokenizer
                
                model_id = "emilianJR/epiCRealism"
                motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-2"
                
                logger.info(f"AnimateDiff 모델 로드 중: {model_id}")
                
                # Motion adapter와 base model 로드
                self.animatediff_model = AnimateDiffPipeline.from_pretrained(
                    model_id,
                    motion_adapter_path=motion_adapter_id,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                )
                
                if self.use_gpu:
                    self.animatediff_model = self.animatediff_model.to(self.device)
                
                logger.info("AnimateDiff 모델 로드 완료")
                
            except (ImportError, AttributeError):
                # 방법 2: 간단한 대체 구현 (Stable Diffusion + 프레임 간 보간)
                logger.warning("AnimateDiff를 직접 로드할 수 없습니다. Stable Diffusion + 보간 방식 사용")
                self._load_stable_diffusion()
                self.animatediff_model = self.stable_diffusion_model
                
        except Exception as e:
            logger.error(f"AnimateDiff 로드 실패: {e}")
            logger.warning("Stable Diffusion으로 대체합니다.")
            self._load_stable_diffusion()
            self.animatediff_model = self.stable_diffusion_model
    
    def _frames_to_video(self, frames: List, output_path: str, fps: int):
        """프레임 리스트를 비디오로 변환"""
        try:
            import imageio
            
            # 프레임을 numpy 배열로 변환
            frame_arrays = []
            for frame in frames:
                if isinstance(frame, torch.Tensor):
                    # Tensor를 numpy로 변환
                    if frame.dim() == 4:  # [B, C, H, W]
                        frame = frame[0]
                    if frame.dim() == 3 and frame.shape[0] == 3:  # [C, H, W]
                        frame = frame.permute(1, 2, 0)  # [H, W, C]
                    frame_np = frame.cpu().numpy()
                    # 정규화 (0-1 범위를 0-255로)
                    if frame_np.max() <= 1.0:
                        frame_np = (frame_np * 255).astype(np.uint8)
                    else:
                        frame_np = frame_np.astype(np.uint8)
                    frame_arrays.append(frame_np)
                elif isinstance(frame, Image.Image):
                    frame_arrays.append(np.array(frame))
                else:
                    frame_arrays.append(np.array(frame))
            
            # 비디오로 저장
            imageio.mimwrite(output_path, frame_arrays, fps=fps, codec='libx264', quality=8)
            
        except ImportError:
            # imageio가 없으면 FFmpeg 사용
            logger.warning("imageio가 없습니다. FFmpeg를 사용합니다.")
            self._frames_to_video_ffmpeg(frames, output_path, fps)
        except Exception as e:
            logger.warning(f"imageio로 비디오 저장 실패: {e}. FFmpeg로 재시도합니다.")
            self._frames_to_video_ffmpeg(frames, output_path, fps)
    
    def _frames_to_video_ffmpeg(self, frames: List, output_path: str, fps: int):
        """FFmpeg를 사용하여 프레임을 비디오로 변환"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 프레임을 이미지로 저장
            frame_paths = []
            for i, frame in enumerate(frames):
                if isinstance(frame, torch.Tensor):
                    frame = Image.fromarray((frame.cpu().numpy() * 255).astype(np.uint8))
                elif not isinstance(frame, Image.Image):
                    frame = Image.fromarray(np.array(frame))
                
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                frame.save(frame_path)
                frame_paths.append(frame_path)
            
            # FFmpeg로 비디오 생성
            cmd = [
                'ffmpeg',
                '-y',
                '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
        finally:
            # 임시 파일 정리
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _combine_video_audio(self, video_path: str, audio_path: str, output_path: str) -> str:
        """비디오와 오디오를 결합"""
        from pathlib import Path
        
        # 입력 파일 존재 확인
        video_exists = Path(video_path).exists()
        audio_exists = Path(audio_path).exists()
        
        if not video_exists:
            logger.error(f"비디오 파일이 존재하지 않습니다: {video_path}")
            raise FileNotFoundError(f"비디오 파일 없음: {video_path}")
        
        if not audio_exists:
            logger.error(f"오디오 파일이 존재하지 않습니다: {audio_path}")
            raise FileNotFoundError(f"오디오 파일 없음: {audio_path}")
        
        # 파일 크기 확인
        video_size = Path(video_path).stat().st_size
        audio_size = Path(audio_path).stat().st_size
        
        if video_size == 0:
            logger.error(f"비디오 파일이 비어있습니다: {video_path}")
            raise ValueError(f"비디오 파일 크기 0: {video_path}")
        
        if audio_size == 0:
            logger.error(f"오디오 파일이 비어있습니다: {audio_path}")
            raise ValueError(f"오디오 파일 크기 0: {audio_path}")
        
        logger.info(f"비디오 파일: {video_path} ({video_size} bytes)")
        logger.info(f"오디오 파일: {audio_path} ({audio_size} bytes)")
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            '-y',
            output_path
        ]
        
        try:
            logger.info("비디오와 오디오 결합 중...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
            logger.info(f"✅ 비디오/오디오 결합 완료: {output_path}")
            return output_path
        except subprocess.TimeoutExpired:
            logger.error("비디오/오디오 결합 시간 초과 (120초)")
            raise
        except subprocess.CalledProcessError as e:
            logger.error("=" * 60)
            logger.error("❌ FFmpeg 비디오/오디오 결합 실패")
            logger.error("=" * 60)
            logger.error(f"명령어: {' '.join(cmd)}")
            logger.error(f"종료 코드: {e.returncode}")
            logger.error(f"표준 출력: {e.stdout}")
            logger.error(f"표준 오류: {e.stderr}")
            logger.error("=" * 60)
            
            # 코덱 문제일 수 있으므로 재인코딩 시도
            logger.warning("코덱 문제일 수 있습니다. 재인코딩을 시도합니다...")
            cmd_reencode = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'libx264',  # 재인코딩
                '-c:a', 'aac',
                '-shortest',
                '-y',
                output_path
            ]
            
            try:
                result = subprocess.run(cmd_reencode, capture_output=True, text=True, check=True, timeout=120)
                logger.info(f"✅ 재인코딩으로 비디오/오디오 결합 완료: {output_path}")
                return output_path
            except Exception as e2:
                logger.error(f"재인코딩도 실패: {e2}")
                raise RuntimeError(f"비디오/오디오 결합 실패: {e.stderr}")
    
    def _generate_with_images(self, text: str, audio_path: str, 
                            output_path: str, duration: float) -> str:
        """이미지 생성 모델을 사용한 비디오 생성 (기존 방식)"""
        logger.warning("기존 이미지 생성 방식은 간단한 슬라이드쇼로 대체됩니다.")
        return self._generate_simple_slideshow(text, audio_path, output_path, duration)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """오디오 파일 길이 반환 (초)"""
        from pathlib import Path
        
        # 파일 존재 확인
        if not Path(audio_path).exists():
            logger.warning(f"오디오 파일이 존재하지 않습니다: {audio_path}")
            return 10.0
        
        # 파일 크기 확인
        file_size = Path(audio_path).stat().st_size
        if file_size == 0:
            logger.warning(f"오디오 파일이 비어있습니다: {audio_path}")
            return 10.0
        
        logger.debug(f"오디오 파일 확인: {audio_path} ({file_size} bytes)")
        
        try:
            # 먼저 ffprobe로 시도
            cmd = [
                'ffprobe',
                '-v', 'error',  # 'quiet' 대신 'error'로 변경하여 오류 메시지 확인
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
            duration_str = result.stdout.strip()
            
            if duration_str:
                duration = float(duration_str)
                logger.debug(f"오디오 길이: {duration:.2f}초")
                return duration
            else:
                logger.warning(f"ffprobe가 오디오 길이를 반환하지 않았습니다.")
                # 대체 방법으로 추정
                return self._estimate_audio_duration(audio_path, file_size)
                
        except subprocess.TimeoutExpired:
            logger.warning("ffprobe 타임아웃, 대체 방법 사용")
            return self._estimate_audio_duration(audio_path, file_size)
        except subprocess.CalledProcessError as e:
            logger.warning(f"ffprobe 실패 (종료 코드: {e.returncode})")
            if e.stderr:
                logger.warning(f"ffprobe 오류: {e.stderr}")
            # 대체 방법으로 추정
            return self._estimate_audio_duration(audio_path, file_size)
        except ValueError as e:
            logger.warning(f"오디오 길이 변환 실패: {e}")
            return self._estimate_audio_duration(audio_path, file_size)
        except Exception as e:
            logger.warning(f"오디오 길이 추출 실패: {e}, 대체 방법 사용")
            return self._estimate_audio_duration(audio_path, file_size)
    
    def _estimate_audio_duration(self, audio_path: str, file_size: int) -> float:
        """파일 크기로 오디오 길이 추정 (대체 방법)"""
        from pathlib import Path
        
        # WAV 파일인 경우 헤더에서 직접 읽기 시도
        if audio_path.lower().endswith('.wav'):
            try:
                import wave
                with wave.open(audio_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    logger.info(f"WAV 파일에서 직접 길이 추출: {duration:.2f}초")
                    return duration
            except Exception as e:
                logger.warning(f"WAV 파일 직접 읽기 실패: {e}")
        
        # 대략적인 추정: WAV 파일 기준 (44.1kHz, 16-bit, stereo)
        # 대략 1초당 176KB (44100 * 2채널 * 2바이트)
        estimated_duration = file_size / (44100 * 2 * 2)
        estimated_duration = max(1.0, min(estimated_duration, 600.0))  # 1~600초 범위
        logger.info(f"오디오 길이 추정 (파일 크기 기반): {estimated_duration:.2f}초")
        return estimated_duration
    
    def _create_silent_audio(self, output_path: str, duration: float):
        """무음 오디오 생성 (오디오 파일이 없거나 손상된 경우 대체)"""
        try:
            import subprocess
            from pathlib import Path
            
            # 최소 0.1초, 최대 600초
            duration = max(0.1, min(duration, 600.0))
            
            # 출력 디렉토리 생성
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=mono:sample_rate=22050',
                '-t', str(duration),
                '-acodec', 'pcm_s16le',  # WAV 포맷 명시
                '-ar', '22050',  # 샘플레이트
                '-ac', '1',  # 모노 채널
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=30
            )
            
            # 파일 생성 확인
            if not Path(output_path).exists():
                raise FileNotFoundError(f"무음 오디오 파일이 생성되지 않았습니다: {output_path}")
            
            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                raise ValueError(f"무음 오디오 파일이 비어있습니다: {output_path}")
            
            logger.debug(f"무음 오디오 생성 성공: {output_path} ({file_size} bytes)")
            
        except subprocess.TimeoutExpired:
            logger.error("무음 오디오 생성 시간 초과")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg 오류: {e.stderr}")
            # 대체 방법: 간단한 WAV 파일 생성
            try:
                import wave
                import struct
                sample_rate = 22050
                num_samples = int(duration * sample_rate)
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 모노
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    # 무음 데이터 (0 값)
                    wav_file.writeframes(b'\x00\x00' * num_samples)
                logger.info(f"대체 방법으로 무음 오디오 생성 완료: {output_path}")
            except Exception as e2:
                logger.error(f"대체 방법도 실패: {e2}")
                raise
        except Exception as e:
            logger.error(f"무음 오디오 생성 실패: {e}")
            raise


class EnglishTTS:
    """영어 TTS 생성기 (외국어 비디오용)"""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: TTS 설정
        """
        self.config = config or {}
        self.config = config or {}
        logger.info("영어 TTS 초기화")
    
    def synthesize(self, text: str, output_path: str, voice: str = None) -> str:
        """
        영어 텍스트를 고품질 미국 남성 내레이션으로 변환
        """
        logger.info(f"영어 TTS 생성 중: {len(text)}자")

        try:
            from TTS.api import TTS
            import re
            from pathlib import Path

            # 남자 예고편 내레이션 톤
            model_name = "tts_models/en/ljspeech/glow-tts"
            vocoder_name = "vocoder_models/en/ljspeech/hifigan_v2"

            logger.info("🎙️ Coqui TTS 모델 로드 (남성 나레이션 톤)...")
            tts = TTS(
                model_name=model_name,
                vocoder_name=vocoder_name,
                progress_bar=False,
                gpu=True
            )

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # 길고 복잡한 텍스트는 ASCII 정리
            clean_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            # 예고편 나레이션 느낌 파라미터
            speed = 0.88          # 더 묵직하게
            noise_scale = 0.25    # 발음 뭉개짐 최소화
            length_scale = 1.05   # 여유 있게 말하기

            logger.info("🎬 음성 생성 시작 (남성 예고편 스타일)...")

            tts.tts_to_file(
                text=clean_text,
                file_path=output_path,
                speed=speed,
                noise_scale=noise_scale,
                length_scale=length_scale
            )

            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                logger.info(f"✅ 영어 TTS 생성 완료: {output_path}")
                return output_path
            else:
                raise RuntimeError("오디오 파일이 비어있습니다.")

        except Exception as e:
            logger.error(f"❌ TTS 생성 실패: {e}")
            duration = max(1.0, len(text) * 0.11)
            self._create_silent_audio(output_path, duration)
            logger.info(f"무음 오디오 생성: {output_path}")
            return output_path
    
    def _create_silent_audio(self, output_path: str, duration: float):
        """무음 오디오 생성 (오디오 파일이 없거나 손상된 경우 대체)"""
        try:
            import subprocess
            from pathlib import Path
            
            # 최소 0.1초, 최대 600초
            duration = max(0.1, min(duration, 600.0))
            
            # 출력 디렉토리 생성
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=mono:sample_rate=22050',
                '-t', str(duration),
                '-acodec', 'pcm_s16le',  # WAV 포맷 명시
                '-ar', '22050',  # 샘플레이트
                '-ac', '1',  # 모노 채널
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=30
            )
            
            # 파일 생성 확인
            if not Path(output_path).exists():
                raise FileNotFoundError(f"무음 오디오 파일이 생성되지 않았습니다: {output_path}")
            
            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                raise ValueError(f"무음 오디오 파일이 비어있습니다: {output_path}")
            
            logger.debug(f"무음 오디오 생성 성공: {output_path} ({file_size} bytes)")
            
        except subprocess.TimeoutExpired:
            logger.error("무음 오디오 생성 시간 초과")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg 오류: {e.stderr}")
            # 대체 방법: 간단한 WAV 파일 생성
            try:
                import wave
                import struct
                sample_rate = 22050
                num_samples = int(duration * sample_rate)
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 모노
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    # 무음 데이터 (0 값)
                    wav_file.writeframes(b'\x00\x00' * num_samples)
                logger.info(f"Python wave 모듈로 무음 오디오 생성: {output_path}")
            except Exception as wave_error:
                logger.error(f"Wave 모듈 오류: {wave_error}")
                raise
        except Exception as e:
            logger.error(f"무음 오디오 생성 실패: {e}")
            raise


# ============================================================
# 한국어 TTS 생성기 (YouTube Shorts용)
# ============================================================
                piper_exe = scripts_dir / "piper.exe"
                if piper_exe.exists():
                    piper_cmd = str(piper_exe)
                else:
                    # 사용자 Scripts 디렉토리
                    user_scripts = Path.home() / "AppData" / "Roaming" / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts"
                    piper_exe = user_scripts / "piper.exe"
                    if piper_exe.exists():
                        piper_cmd = str(piper_exe)
            
            # 방법 1: Python API 사용 (권장)
            try:
                from piper import PiperVoice
                import json
                
                # 모델 다운로드 및 경로 찾기
                model_path = None
                config_path = None
                
                # HuggingFace 캐시에서 모델 찾기 (재귀적 검색)
                hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                hf_model_dir = hf_cache_dir / "models--rhasspy--piper-voices"
                hf_voice_paths = []
                
                if hf_model_dir.exists():
                    logger.info(f"✅ HuggingFace 캐시 발견: {hf_model_dir}")
                    # snapshots 디렉토리에서 최신 스냅샷 찾기
                    snapshots_dir = hf_model_dir / "snapshots"
                    if snapshots_dir.exists():
                        snapshots = list(snapshots_dir.iterdir())
                        if snapshots:
                            # 모든 스냅샷에서 검색
                            for snapshot in snapshots:
                                logger.debug(f"스냅샷 확인: {snapshot.name}")
                                # 재귀적으로 en_US-amy-medium.onnx 파일 찾기
                                for onnx_file in snapshot.rglob(f"{voice}.onnx"):
                                    logger.info(f"✅ 모델 파일 발견: {onnx_file}")
                                    hf_voice_paths.append(onnx_file.parent)
                                    break
                                
                                # 표준 경로도 추가
                                hf_voice_paths.append(snapshot / "en" / "en_US" / "amy" / "medium")
                                hf_voice_paths.append(snapshot / "en" / "en_US")
                                hf_voice_paths.append(snapshot / "en")
                                hf_voice_paths.append(snapshot)
                            
                            if hf_voice_paths:
                                latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                                logger.info(f"최신 스냅샷: {latest_snapshot.name}")
                else:
                    logger.warning(f"⚠️  HuggingFace 캐시 없음: {hf_model_dir}")
                    logger.info("모델 다운로드 필요: python download_all_models.py --auto")
                
                # download_all_models.py가 저장하는 경로 (최우선)
                download_save_dir = Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium"
                
                # 가능한 모델 경로들 (download_all_models.py 저장 경로 우선, 그 다음 HuggingFace 캐시)
                possible_dirs = [download_save_dir] + hf_voice_paths + [
                    Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium",
                    Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US",
                    Path.home() / ".local" / "share" / "piper" / "voices" / "en",
                    Path.home() / ".local" / "share" / "piper" / "voices",
                    Path("models") / "tts",
                    Path.home() / "AppData" / "Local" / "piper" / "voices",
                    Path.home() / "AppData" / "Local" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium",
                ]
                
                logger.info(f"download_all_models.py 저장 경로 확인: {download_save_dir}")
                if download_save_dir.exists():
                    logger.info(f"✅ 저장 경로 존재: {download_save_dir}")
                    # 해당 디렉토리의 모든 .onnx 파일 확인
                    onnx_files = list(download_save_dir.glob("*.onnx"))
                    if onnx_files:
                        logger.info(f"  발견된 .onnx 파일: {[f.name for f in onnx_files]}")
                else:
                    logger.warning(f"⚠️  저장 경로 없음: {download_save_dir}")
                
                # 음성 이름 정규화 (en_US-amy-medium -> 그대로 사용)
                voice_name_normalized = voice  # en_US-amy-medium 그대로 사용
                
                # 먼저 정확한 파일명으로 찾기
                logger.info(f"모델 검색 시작: {voice}")
                logger.info(f"검색 경로 개수: {len(possible_dirs)}")
                
                for base_dir in possible_dirs:
                    if not base_dir.exists():
                        continue
                    
                    # 재귀적으로 파일 찾기
                    try:
                        for onnx_file in base_dir.rglob(f"{voice}.onnx"):
                            logger.info(f"✅ 모델 파일 발견 (재귀 검색): {onnx_file}")
                            model_path = onnx_file
                            config_path = onnx_file.with_suffix(".onnx.json")
                            if not config_path.exists():
                                # 같은 디렉토리에서 config 찾기
                                config_candidates = [
                                    onnx_file.parent / f"{voice}.onnx.json",
                                    onnx_file.parent / "model.onnx.json",
                                ]
                                for candidate in config_candidates:
                                    if candidate.exists():
                                        config_path = candidate
                                        break
                            logger.info(f"✅ 모델: {model_path}")
                            if config_path and config_path.exists():
                                logger.info(f"✅ Config: {config_path}")
                            break
                    except Exception as e:
                        logger.debug(f"재귀 검색 오류 ({base_dir}): {e}")
                    
                    if model_path:
                        break
                    
                    # 정확한 파일명 패턴 (우선순위 높음)
                    exact_patterns = [
                        base_dir / f"{voice}.onnx",  # en_US-amy-medium.onnx
                    ]
                    
                    for pattern in exact_patterns:
                        if pattern.exists():
                            model_path = pattern
                            config_path = pattern.with_suffix(".onnx.json")
                            if not config_path.exists():
                                # 같은 디렉토리에서 config 찾기
                                config_candidates = [
                                    base_dir / f"{voice}.onnx.json",
                                    base_dir / "model.onnx.json",
                                ]
                                for candidate in config_candidates:
                                    if candidate.exists():
                                        config_path = candidate
                                        break
                            logger.info(f"✅ 모델 발견: {model_path}")
                            if config_path and config_path.exists():
                                logger.info(f"✅ Config 발견: {config_path}")
                            break
                    
                    if model_path:
                        break
                
                # 정확한 파일명으로 못 찾았으면 대체 패턴 시도
                if not model_path:
                    for base_dir in possible_dirs:
                        if not base_dir.exists():
                            continue
                        
                        # 대체 패턴
                        fallback_patterns = [
                            base_dir / "model.onnx",  # 일부는 model.onnx로 저장될 수 있음
                            base_dir / "model.onnx.json",
                            base_dir.parent / f"{voice}.onnx",  # 상위 디렉토리도 확인
                            base_dir.parent / f"{voice}.onnx.json",
                        ]
                        
                        for pattern in fallback_patterns:
                            if pattern.exists() and pattern.suffix == ".onnx":
                                model_path = pattern
                                config_path = pattern.with_suffix(".onnx.json")
                                if not config_path.exists():
                                    config_path = None
                                logger.info(f"✅ 모델 발견 (대체): {model_path}")
                                break
                        
                        if model_path:
                            break
                
                if not model_path:
                    # 모델이 없으면 다운로드 시도
                    logger.warning(f"음성 모델 '{voice}'를 찾을 수 없습니다.")
                    logger.info("모델 다운로드를 시도합니다...")
                    
                    # piper 명령어로 다운로드 시도
                    if piper_cmd:
                        try:
                            download_cmd = [piper_cmd, 'download', '--voice', voice]
                            result = subprocess.run(
                                download_cmd,
                                text=True,
                                capture_output=True,
                                check=False
                            )
                            if result.returncode == 0:
                                logger.info("모델 다운로드 완료, 다시 시도합니다...")
                                # 다운로드 후 다시 경로 찾기
                                for base_dir in possible_dirs:
                                    pattern = base_dir / f"{voice}.onnx"
                                    if pattern.exists():
                                        model_path = pattern
                                        config_path = pattern.with_suffix(".onnx.json")
                                        break
                        except Exception as dl_error:
                            logger.warning(f"모델 다운로드 실패: {dl_error}")
                    
                    if not model_path:
                        logger.warning(f"⚠️ Piper 음성 모델 '{voice}'를 찾을 수 없습니다.")
                        logger.warning("무음 오디오로 대체합니다. (실제 TTS를 사용하려면 모델 다운로드 필요)")
                        logger.info("모델 다운로드 방법:")
                        logger.info("  python -m piper.download --voice en_US-lessac-medium")
                        logger.info("  또는 https://huggingface.co/rhasspy/piper-voices 에서 수동 다운로드")
                        # 무음 오디오로 대체
                        duration = max(1.0, len(text) * 0.1)
                        self._create_silent_audio(output_path, duration)
                        logger.info(f"무음 오디오 생성 완료: {output_path} (길이: {duration:.1f}초)")
                        return output_path
                
                logger.info(f"Piper Python API 사용: {model_path}")
                model = PiperVoice.load(model_path, config_path)
                
                # 텍스트 정리 및 길이 제한
                MAX_TEXT_LENGTH = 1000  # Piper 안전 최대 길이
                
                # 텍스트 정리 (Piper가 처리하기 어려운 문자 제거)
                import re
                # 기본 영문자, 숫자, 구두점만 유지
                clean_text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # ASCII 문자만 유지
                clean_text = re.sub(r'\s+', ' ', clean_text)  # 연속된 공백 정리
                clean_text = clean_text.strip()
                
                if len(clean_text) > MAX_TEXT_LENGTH:
                    logger.warning(f"텍스트가 너무 깁니다 ({len(clean_text)}자). {MAX_TEXT_LENGTH}자로 자릅니다.")
                    clean_text = clean_text[:MAX_TEXT_LENGTH]
                
                if not clean_text:
                    logger.error("텍스트 정리 후 빈 문자열입니다.")
                    logger.warning("무음 오디오로 대체합니다...")
                    duration = max(1.0, len(text) * 0.1)
                    self._create_silent_audio(output_path, duration)
                    return output_path
                
                logger.debug(f"원본 텍스트 길이: {len(text)}자, 정리 후: {len(clean_text)}자")
                
                # 오디오 생성
                logger.info(f"Piper로 오디오 합성 중... (텍스트 길이: {len(clean_text)}자)")
                
                # 짧은 테스트 텍스트로 먼저 시도
                test_text = "Hello world. This is a test."
                logger.debug(f"Piper 테스트 시작: '{test_text}'")
                
                try:
                    # 실제 합성 시도
                    with open(output_path, "wb") as f:
                        # 테스트: 짧은 텍스트로 먼저 시도
                        logger.debug("Piper synthesize() 호출 시작...")
                        model.synthesize(clean_text, f)
                        f.flush()  # 버퍼 플러시
                        logger.debug("Piper synthesize() 호출 완료")
                    
                    # 즉시 파일 크기 확인
                    import os
                    if os.path.exists(output_path):
                        temp_size = os.path.getsize(output_path)
                        logger.debug(f"합성 직후 파일 크기: {temp_size} bytes")
                    else:
                        logger.error("합성 후 파일이 존재하지 않음!")
                        
                except Exception as synth_error:
                    logger.error(f"Piper 합성 중 오류: {synth_error}")
                    import traceback
                    logger.error(f"상세 오류:\n{traceback.format_exc()}")
                    logger.warning("무음 오디오로 대체합니다...")
                    duration = max(1.0, len(text) * 0.1)
                    self._create_silent_audio(output_path, duration)
                    logger.info(f"무음 오디오 생성 완료: {output_path} (길이: {duration:.1f}초)")
                    return output_path
                
                # 파일 생성 확인
                from pathlib import Path
                output_file = Path(output_path)
                if not output_file.exists():
                    logger.error(f"오디오 파일이 생성되지 않았습니다: {output_path}")
                    logger.warning("무음 오디오로 대체합니다...")
                    duration = max(1.0, len(text) * 0.1)
                    self._create_silent_audio(output_path, duration)
                    return output_path
                
                file_size = output_file.stat().st_size
                if file_size == 0:
                    logger.error(f"오디오 파일이 비어있습니다 (0 bytes): {output_path}")
                    logger.error("⚠️  Piper 모델에 문제가 있습니다!")
                    logger.warning("고품질 대체 TTS 엔진을 시도합니다...")
                    
                    # 대체 1: Coqui TTS 시도 (가장 자연스러움, 오픈소스)
                    try:
                        logger.info("🎙️  Coqui TTS (고품질, 자연스러움) 시도 중...")
                        
                        # espeak-ng 경로를 환경 변수에 직접 설정
                        import os
                        espeak_paths = [
                            r"C:\Program Files\eSpeak NG",
                            r"C:\Program Files (x86)\eSpeak NG",
                            r"C:\ProgramData\chocolatey\lib\espeak-ng\tools"
                        ]
                        
                        # espeak-ng.exe 찾기
                        espeak_found = False
                        for path in espeak_paths:
                            espeak_exe = os.path.join(path, "espeak-ng.exe")
                            if os.path.exists(espeak_exe):
                                # 환경 변수 설정
                                os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_exe
                                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(path, "libespeak-ng.dll") if os.path.exists(os.path.join(path, "libespeak-ng.dll")) else espeak_exe
                                # PATH에도 추가
                                if path not in os.environ.get("PATH", ""):
                                    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                                logger.info(f"✅ espeak-ng 찾음: {espeak_exe}")
                                espeak_found = True
                                break
                        
                        if not espeak_found:
                            logger.warning("⚠️  espeak-ng를 찾을 수 없습니다. 다음 경로 확인:")
                            for path in espeak_paths:
                                logger.warning(f"   - {path}")
                            raise ImportError("espeak-ng not found")
                        
                        from TTS.api import TTS
                        
                        # Coqui TTS 초기화 (남성 목소리 모델)
                        # VCTK 모델: 여러 화자 지원, 남성 목소리 선택 가능
                        tts_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)
                        
                        # 영화 나레이션용 깊고 권위있는 남성 화자 선택:
                        # p259: 남성, 깊고 진지한 목소리 (영화 나레이션 스타일) 🎬 ← 강력 추천!
                        # p274: 남성, 낮고 권위있는 목소리
                        # p260: 남성, 성숙하고 안정적인 목소리
                        # p232: 남성, 나이든 목소리 (할아버지 느낌)
                        # p243: 남성, 중년의 전문적인 목소리
                        tts_model.tts_to_file(text=clean_text, file_path=output_path, speaker="p243")
                        
                        # 파일 생성 확인
                        if output_file.exists() and output_file.stat().st_size > 0:
                            logger.info(f"✅ Coqui TTS로 고품질 오디오 생성 완료: {output_path} ({output_file.stat().st_size} bytes)")
                            return output_path
                        else:
                            logger.warning("Coqui TTS도 빈 파일을 생성했습니다.")
                    except ImportError:
                        logger.warning("Coqui TTS가 설치되지 않았습니다. 설치: pip install TTS")
                    except Exception as coqui_error:
                        logger.warning(f"Coqui TTS 실패: {coqui_error}")
                    
                    # 대체 2: edge-tts 시도 (매우 자연스러움)
                    try:
                        logger.info("🎙️  edge-tts (Microsoft Edge TTS, 매우 자연스러움) 시도 중...")
                        import asyncio
                        import edge_tts
                        
                        # edge-tts는 긴 텍스트를 처리할 수 있지만, 500자 이하로 나누면 더 안정적
                        max_chunk_length = 500
                        
                        async def generate_edge_tts():
                            # 영화 나레이션 스타일 남성 목소리
                            # DavisNeural: 남성, 나레이션/다큐멘터리 스타일 (영화 같은 느낌) 🎬
                            # GuyNeural: 남성, 자연스러운 일반 목소리
                            # TonyNeural: 남성, 뉴스캐스터 스타일
                            
                            # 텍스트가 너무 길면 청크로 나누기
                            if len(clean_text) > max_chunk_length:
                                logger.info(f"텍스트를 {max_chunk_length}자 청크로 나눕니다...")
                                # 문장 단위로 나누기
                                sentences = clean_text.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
                                
                                chunks = []
                                current_chunk = ""
                                for sentence in sentences:
                                    if len(current_chunk) + len(sentence) < max_chunk_length:
                                        current_chunk += sentence + " "
                                    else:
                                        if current_chunk:
                                            chunks.append(current_chunk.strip())
                                        current_chunk = sentence + " "
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                
                                # 각 청크를 임시 파일로 생성
                                import tempfile
                                temp_files = []
                                for i, chunk in enumerate(chunks):
                                    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
                                    temp_file.close()
                                    communicate = edge_tts.Communicate(chunk, "en-US-DavisNeural")
                                    await communicate.save(temp_file.name)
                                    temp_files.append(temp_file.name)
                                
                                # ffmpeg로 파일 합치기
                                if len(temp_files) > 1:
                                    import subprocess
                                    # 파일 리스트 생성
                                    list_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
                                    for tf in temp_files:
                                        list_file.write(f"file '{tf}'\n")
                                    list_file.close()
                                    
                                    # ffmpeg로 합치기
                                    subprocess.run([
                                        'ffmpeg', '-f', 'concat', '-safe', '0',
                                        '-i', list_file.name, '-c', 'copy', '-y', output_path
                                    ], check=True, capture_output=True)
                                    
                                    # 임시 파일 삭제
                                    import os
                                    os.unlink(list_file.name)
                                    for tf in temp_files:
                                        os.unlink(tf)
                                else:
                                    # 파일이 하나면 그냥 이동
                                    import shutil
                                    shutil.move(temp_files[0], output_path)
                            else:
                                # 텍스트가 짧으면 바로 처리
                                communicate = edge_tts.Communicate(clean_text, "en-US-DavisNeural")
                                await communicate.save(output_path)
                        
                        asyncio.run(generate_edge_tts())
                        
                        # 파일 생성 확인
                        if output_file.exists() and output_file.stat().st_size > 0:
                            logger.info(f"✅ edge-tts로 고품질 오디오 생성 완료: {output_path} ({output_file.stat().st_size} bytes)")
                            return output_path
                        else:
                            logger.warning("edge-tts도 빈 파일을 생성했습니다.")
                    except ImportError:
                        logger.warning("edge-tts가 설치되지 않았습니다. 설치: pip install edge-tts")
                    except Exception as edge_error:
                        logger.warning(f"edge-tts 실패: {edge_error}")
                        import traceback
                        logger.debug(f"edge-tts 상세 오류:\n{traceback.format_exc()}")
                    
                    # 대체 3: gTTS 시도 (간단하고 빠름)
                    try:
                        logger.info("🎙️  gTTS (Google Text-to-Speech) 시도 중...")
                        from gtts import gTTS
                        tts = gTTS(text=clean_text, lang='en', slow=False)
                        tts.save(output_path)
                        
                        # 파일 생성 확인
                        if output_file.exists() and output_file.stat().st_size > 0:
                            logger.info(f"✅ gTTS로 오디오 생성 완료: {output_path} ({output_file.stat().st_size} bytes)")
                            return output_path
                        else:
                            logger.warning("gTTS도 빈 파일을 생성했습니다.")
                    except ImportError:
                        logger.warning("gTTS가 설치되지 않았습니다. 설치: pip install gtts")
                    except Exception as gtts_error:
                        logger.warning(f"gTTS 실패: {gtts_error}")
                    
                    # 최종 대체: 무음 오디오
                    logger.warning("모든 TTS 엔진이 실패했습니다. 무음 오디오로 대체합니다...")
                    duration = max(1.0, len(text) * 0.1)
                    self._create_silent_audio(output_path, duration)
                    logger.info(f"무음 오디오 생성 완료: {output_path} (길이: {duration:.1f}초)")
                    return output_path
                
                logger.info(f"✅ 영어 TTS 생성 완료: {output_path} ({file_size} bytes)")
                return output_path
                
            except (ImportError, FileNotFoundError) as api_error:
                # 방법 2: 명령줄 도구 사용
                if piper_cmd:
                    logger.info(f"Piper 명령줄 도구 사용: {piper_cmd}")
                    # Piper는 stdin에서 텍스트를 읽지 않고, --text 옵션 사용
                    cmd = [piper_cmd, '--model', voice, '--output_file', output_path, '--text', text]
                    result = subprocess.run(
                        cmd, 
                        text=True, 
                        capture_output=True, 
                        check=False  # check=False로 변경하여 오류 메시지 확인
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"영어 TTS 생성 완료: {output_path}")
                        return output_path
                    else:
                        error_msg = result.stderr or result.stdout
                        logger.error(f"Piper 명령어 실패: {error_msg}")
                        raise RuntimeError(f"Piper TTS 실패: {error_msg}")
                else:
                    raise FileNotFoundError("Piper를 사용할 수 없습니다. 모델을 다운로드하거나 piper 명령어를 설치하세요.")
                    
        except Exception as e:
            logger.error(f"영어 TTS 생성 실패: {e}")
            raise
                    
        except Exception as e:
            logger.error(f"영어 TTS 생성 실패: {e}")
            logger.warning("대체 방법: 텍스트 길이에 맞춰 무음 오디오 생성")
            # 대체: 텍스트 길이에 맞춰 무음 오디오 생성 (대략 1초에 10자)
            duration = max(1.0, len(text) * 0.1)
            self._create_silent_audio(output_path, duration)
            logger.info(f"무음 오디오 생성 완료: {output_path} (길이: {duration:.1f}초)")
            return output_path
    
    def _create_silent_audio(self, output_path: str, duration: float):
        """무음 오디오 생성 (대체)"""
        try:
            import subprocess
            from pathlib import Path
            
            # 최소 0.1초, 최대 600초
            duration = max(0.1, min(duration, 600.0))
            
            # 출력 디렉토리 생성
            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=mono:sample_rate=22050',
                '-t', str(duration),
                '-acodec', 'pcm_s16le',  # WAV 포맷 명시
                '-ar', '22050',  # 샘플레이트
                '-ac', '1',  # 모노 채널
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=30
            )
            
            # 파일 생성 확인
            if not Path(output_path).exists():
                raise FileNotFoundError(f"무음 오디오 파일이 생성되지 않았습니다: {output_path}")
            
            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                raise ValueError(f"무음 오디오 파일이 비어있습니다: {output_path}")
            
            logger.debug(f"무음 오디오 생성 성공: {output_path} ({file_size} bytes)")
            
        except subprocess.TimeoutExpired:
            logger.error("무음 오디오 생성 시간 초과")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg 오류: {e.stderr}")
            # 대체 방법: 간단한 WAV 파일 생성
            try:
                import wave
                import struct
                sample_rate = 22050
                num_samples = int(duration * sample_rate)
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 모노
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    # 무음 데이터 (0 값)
                    wav_file.writeframes(b'\x00\x00' * num_samples)
                logger.info(f"대체 방법으로 무음 오디오 생성 완료: {output_path}")
            except Exception as e2:
                logger.error(f"대체 방법도 실패: {e2}")
                raise
        except Exception as e:
            logger.error(f"무음 오디오 생성 실패: {e}")
            raise


def create_video_generator(config: dict) -> VideoGenerator:
    """설정에서 VideoGenerator 인스턴스 생성"""
    return VideoGenerator(config)


def create_english_tts(config: dict) -> EnglishTTS:
    """설정에서 EnglishTTS 인스턴스 생성"""
    return EnglishTTS(config)

