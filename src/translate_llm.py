"""로컬 LLM을 사용한 영어→한국어 번역 모듈"""

import logging
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
import platform

logger = logging.getLogger(__name__)


class LLMTranslator:
    """로컬 LLM을 사용한 번역기"""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None,
                 use_gpu: bool = True, use_llama_cpp: bool = False,
                 batch_size: int = 4, max_tokens: int = 2048):
        """
        Args:
            model_name: "deepseek-r1-7b" 또는 "llama-3.1-8b"
            model_path: 로컬 모델 경로 (None이면 HuggingFace에서 다운로드)
            use_gpu: GPU 사용 여부
            use_llama_cpp: llama.cpp 사용 여부 (CPU fallback)
            batch_size: 배치 크기
            max_tokens: 최대 토큰 수
        """
        self.model_name = model_name
        self.model_path = model_path
        
        # GPU 사용 설정 (강제 모드 지원)
        if use_gpu:
            if torch.cuda.is_available():
                self.use_gpu = True
                self.device = "cuda"
            else:
                logger.warning("⚠️  GPU 사용 요청했지만 CUDA를 사용할 수 없습니다. CPU 모드로 진행합니다.")
                self.use_gpu = False
                self.device = "cpu"
        else:
            self.use_gpu = False
            self.device = "cpu"
        
        self.use_llama_cpp = use_llama_cpp
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"LLM 초기화: 모델={model_name}, 디바이스={self.device}, GPU={self.use_gpu}")
        if self.use_gpu:
            logger.info(f"GPU 정보: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        
        # 모델 HuggingFace ID 매핑
        self.model_ids = {
            "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1",
            "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct"
        }
        
        logger.info(f"LLM 번역기 초기화: 모델={model_name}, 디바이스={self.device}")
    
    def load_model(self):
        """모델 및 토크나이저 로드"""
        if self.model is not None:
            return
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_id = self.model_ids.get(self.model_name)
            if model_id is None:
                raise ValueError(f"지원하지 않는 모델: {self.model_name}")
            
            if self.model_path:
                model_id = self.model_path
            
            logger.info(f"모델 로딩 중: {model_id}")
            
            # HuggingFace 캐시 확인
            import os
            from huggingface_hub import snapshot_download
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            
            # 캐시에 모델이 있는지 확인
            if not self.model_path:  # HuggingFace ID를 사용하는 경우
                # 캐시 디렉토리에서 모델 찾기
                model_name_safe = model_id.replace("/", "--")
                cache_model_dir = os.path.join(cache_dir, f"models--{model_name_safe}")
                if os.path.exists(cache_model_dir):
                    logger.info(f"✅ HuggingFace 캐시에서 모델 발견: {cache_model_dir}")
                    logger.info("캐시에서 모델을 로드합니다 (다운로드 불필요)")
                else:
                    logger.info("⚠️  캐시에 모델이 없습니다. 첫 다운로드 시 시간이 오래 걸릴 수 있습니다.")
            
            # HuggingFace Hub 다운로드 설정 개선
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # 타임아웃 10분으로 증가
            os.environ["HF_HUB_DOWNLOAD_MAX_WORKERS"] = "1"  # 동시 다운로드 수를 1개로 제한
            
            # 토크나이저 로드 (캐시 확인 자동)
            logger.info("토크나이저 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                resume_download=True,  # 다운로드 재개 지원
            )
            
            # 모델 로드 (캐시 확인 자동)
            logger.info("모델 파일 로드 중... (캐시에 있으면 다운로드하지 않습니다)")
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.use_gpu else torch.float32,
                "resume_download": True,  # 다운로드 재개 지원
            }
            
            # accelerate가 설치되어 있고 GPU를 사용하는 경우에만 device_map 사용
            use_device_map = False
            if self.use_gpu:
                try:
                    import accelerate
                    use_device_map = True
                    model_kwargs["device_map"] = "auto"
                    logger.info("accelerate를 사용하여 자동 디바이스 매핑 활성화")
                except ImportError:
                    logger.warning("accelerate가 설치되지 않았습니다. device_map을 사용하지 않고 직접 디바이스로 이동합니다.")
                    use_device_map = False
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            # device_map을 사용하지 않은 경우 직접 디바이스로 이동
            if not use_device_map:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info("모델 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            if self.use_llama_cpp:
                logger.info("llama.cpp로 fallback 시도...")
                self._load_llama_cpp()
            else:
                raise
    
    def _load_llama_cpp(self):
        """llama.cpp를 사용한 CPU fallback"""
        try:
            from llama_cpp import Llama
            
            if not self.model_path:
                raise ValueError("llama.cpp 사용 시 model_path가 필요합니다")
            
            logger.info(f"llama.cpp로 모델 로드: {self.model_path}")
            
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=4,
                verbose=False
            )
            self.use_llama_cpp = True
            
        except ImportError:
            logger.error("llama-cpp-python이 설치되지 않았습니다")
            raise
        except Exception as e:
            logger.error(f"llama.cpp 로드 실패: {e}")
            raise
    
    def _create_translation_prompt(self, english_text: str, translation_style: Optional[dict] = None) -> str:
        """번역 프롬프트 생성"""
        if translation_style is None:
            translation_style = {}
        
        tone = translation_style.get("tone", "뉴스 해설자")
        style = translation_style.get("style", "간결하고 명확한 설명체")
        remove_fillers = translation_style.get("remove_fillers", True)
        
        # 톤별 기본 가이드라인
        tone_guides = {
            "뉴스 해설자": "- 객관적이고 사실 중심의 설명\n- 명확하고 간결한 문장\n- 감정 표현 최소화",
            "스토리텔러": "- 서사적이고 몰입감 있는 문체\n- 자연스러운 흐름과 리듬\n- 독자의 상상력을 자극하는 표현",
            "교육자": "- 이해하기 쉽고 명확한 설명\n- 단계별 논리적 구성\n- 중요한 개념 강조\n- 유아용의 경우: 매우 간단한 단어 사용, 짧은 문장, 반복적인 설명, 친근하고 따뜻한 표현",
        }
        
        tone_guide = tone_guides.get(tone, tone_guides["뉴스 해설자"])
        
        filler_instruction = ""
        if remove_fillers:
            filler_instruction = "- 불필요한 접속사나 채움말 제거\n"
        
        prompt = f"""다음 영어 텍스트를 한국어로 번역해주세요. 
번역 시 다음 스타일을 유지해주세요:

톤: {tone}
스타일: {style}

가이드라인:
{tone_guide}
{filler_instruction}- 직역하지 말고 의미를 정확히 전달하되 자연스러운 한국어로 재작성
- 외국 사건을 한국 사람에게 자연스럽게 설명

영어 텍스트:
{english_text}

한국어 번역:"""
        return prompt
    
    def translate_text(self, text: str, translation_style: Optional[dict] = None) -> str:
        """단일 텍스트 번역"""
        if self.model is None:
            self.load_model()
        
        prompt = self._create_translation_prompt(text, translation_style)
        
        try:
            if self.use_llama_cpp:
                return self._translate_with_llama_cpp(prompt)
            else:
                return self._translate_with_transformers(prompt)
        except Exception as e:
            logger.error(f"번역 실패: {e}")
            raise
    
    def _translate_with_transformers(self, prompt: str) -> str:
        """Transformers를 사용한 번역"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 프롬프트 부분 제거
        translation = generated_text[len(prompt):].strip()
        return translation
    
    def _translate_with_llama_cpp(self, prompt: str) -> str:
        """llama.cpp를 사용한 번역"""
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=0.7,
            stop=["영어 텍스트:", "\n\n"],
            echo=False
        )
        return response["choices"][0]["text"].strip()
    
    def translate_segments(self, segments: List[Dict[str, Any]], 
                          translation_style: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        세그먼트 리스트 번역 (타임코드 유지)
        
        Args:
            segments: Whisper에서 반환된 세그먼트 리스트
            translation_style: 번역 스타일 설정 (채널 프로필에서 전달)
            
        Returns:
            번역된 세그먼트 리스트 (타임코드 유지)
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"{len(segments)}개 세그먼트 번역 시작")
        if translation_style:
            logger.info(f"번역 스타일: {translation_style.get('tone', '기본')} - {translation_style.get('style', '')}")
        
        translated_segments = []
        
        for i, segment in enumerate(segments):
            english_text = segment["text"]
            logger.debug(f"세그먼트 {i+1}/{len(segments)} 번역 중...")
            
            try:
                korean_text = self.translate_text(english_text, translation_style)
                
                translated_segment = {
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text_en": english_text,
                    "text_ko": korean_text,
                    "words": segment.get("words", [])
                }
                
                translated_segments.append(translated_segment)
                
            except Exception as e:
                logger.error(f"세그먼트 {i+1} 번역 실패: {e}")
                # 실패 시 원문 유지
                translated_segment = {
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text_en": english_text,
                    "text_ko": english_text,  # fallback
                    "words": segment.get("words", [])
                }
                translated_segments.append(translated_segment)
        
        logger.info("번역 완료")
        return translated_segments


def create_llm_translator(config: dict) -> LLMTranslator:
    """설정에서 LLMTranslator 인스턴스 생성"""
    llm_config = config.get("llm", {})
    return LLMTranslator(
        model_name=llm_config.get("model", "deepseek-r1-7b"),
        model_path=llm_config.get("model_path"),
        use_gpu=llm_config.get("use_gpu", True),
        use_llama_cpp=llm_config.get("use_llama_cpp", False),
        batch_size=llm_config.get("batch_size", 4),
        max_tokens=llm_config.get("max_tokens", 2048)
    )

