"""
8단계 파이프라인 검증 스크립트
각 단계별로 필요한 컴포넌트를 확인하고 문제를 진단합니다.
"""
import sys
import logging
from pathlib import Path
import importlib

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def validate_step(step_num, step_name, check_func):
    """단계별 검증"""
    logger.info(f"\n{'='*60}")
    logger.info(f"[{step_num}/8] {step_name} 검증")
    logger.info(f"{'='*60}")
    
    try:
        result = check_func()
        if result:
            logger.info(f"✅ [{step_num}/8] {step_name} 준비 완료\n")
            return True
        else:
            logger.error(f"❌ [{step_num}/8] {step_name} 문제 발견\n")
            return False
    except Exception as e:
        logger.error(f"❌ [{step_num}/8] {step_name} 검증 오류: {e}\n")
        return False

def check_step1_english_tts():
    """1단계: 영어 TTS 생성"""
    # Piper TTS 확인
    try:
        from piper import PiperVoice
        logger.info("  ✅ Piper TTS 라이브러리 확인")
    except ImportError:
        logger.error("  ❌ Piper TTS 라이브러리 없음")
        return False
    
    # 모델 확인
    voice_name = "en_US-amy-medium"
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    hf_model_dir = hf_cache / "models--rhasspy--piper-voices"
    found = False
    
    if hf_model_dir.exists():
        snapshots_dir = hf_model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                for onnx_file in snapshot.rglob(f"{voice_name}.onnx"):
                    logger.info(f"  ✅ 모델 발견: {onnx_file}")
                    found = True
                    break
                if found:
                    break
    
    if not found:
        save_dir = Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium"
        if (save_dir / f"{voice_name}.onnx").exists():
            logger.info(f"  ✅ 모델 발견: {save_dir}")
            found = True
    
    if not found:
        logger.error("  ❌ 영어 TTS 모델 없음")
        logger.info("  해결: python download_all_models.py --auto")
        return False
    
    return True

def check_step2_video_generation():
    """2단계: 영어 비디오 생성"""
    # GPU 확인
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("  ❌ CUDA를 사용할 수 없습니다.")
            return False
        
        # GPU 테스트
        test = torch.zeros(1).cuda()
        del test
        torch.cuda.empty_cache()
        logger.info("  ✅ GPU 사용 가능")
    except RuntimeError as e:
        error_str = str(e).lower()
        if "kernel image" in error_str or "no kernel" in error_str:
            logger.error("  ❌ GPU Compute Capability 호환성 문제")
            logger.info("  해결: install_pytorch_nightly.bat 실행")
            return False
        else:
            raise
    except ImportError:
        logger.error("  ❌ PyTorch 없음")
        return False
    
    # Diffusers 확인
    try:
        import diffusers
        logger.info(f"  ✅ Diffusers: {diffusers.__version__}")
    except ImportError:
        logger.error("  ❌ Diffusers 없음")
        return False
    
    # 모델 확인 (선택사항 - 첫 실행 시 자동 다운로드)
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    sdxl_dir = hf_cache / "models--stabilityai--stable-diffusion-xl-base-1.0"
    if sdxl_dir.exists():
        logger.info("  ✅ Stable Diffusion XL 모델 발견")
    else:
        logger.info("  ⚠️  Stable Diffusion XL 모델 없음 (첫 실행 시 자동 다운로드)")
    
    return True

def check_step3_stt():
    """3단계: STT 처리"""
    try:
        import whisper
        logger.info("  ✅ Whisper 라이브러리 확인")
    except ImportError:
        logger.error("  ❌ Whisper 라이브러리 없음")
        return False
    
    # GPU 확인 (이미 위에서 확인했지만)
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("  ✅ GPU 사용 가능 (Whisper)")
        else:
            logger.warning("  ⚠️  CPU 모드 (느림)")
    except:
        pass
    
    return True

def check_step4_translation():
    """4단계: 번역"""
    try:
        import transformers
        logger.info(f"  ✅ Transformers: {transformers.__version__}")
    except ImportError:
        logger.error("  ❌ Transformers 없음")
        return False
    
    # LLM 모델 확인 (선택사항 - 첫 실행 시 자동 다운로드)
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    llama_dir = hf_cache / "models--meta-llama--Llama-3.1-8B-Instruct"
    if llama_dir.exists():
        logger.info("  ✅ Llama 모델 발견")
    else:
        logger.info("  ⚠️  Llama 모델 없음 (첫 실행 시 자동 다운로드)")
    
    return True

def check_step5_korean_tts():
    """5단계: 한국어 TTS 생성"""
    # VibeVoice 또는 Piper 확인
    try:
        import torch
        logger.info("  ✅ PyTorch 확인 (VibeVoice용)")
    except ImportError:
        logger.error("  ❌ PyTorch 없음")
        return False
    
    # 한국어 TTS 모델 확인
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    korean_model = hf_cache / "models--neurlang--piper-onnx-kss-korean"
    vibevoice_model = hf_cache / "models--microsoft--VibeVoice-1.5B"
    
    if korean_model.exists():
        logger.info("  ✅ 한국어 Piper 모델 발견")
        return True
    elif vibevoice_model.exists():
        logger.info("  ✅ VibeVoice 모델 발견")
        return True
    else:
        logger.info("  ⚠️  한국어 TTS 모델 없음 (첫 실행 시 자동 다운로드)")
        return True  # 선택사항

def check_step6_subtitles():
    """6단계: 자막 생성"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        logger.info("  ✅ PIL (Pillow) 확인")
    except ImportError:
        logger.error("  ❌ Pillow 없음")
        return False
    
    return True

def check_step7_video_mute():
    """7단계: 비디오 음소거"""
    # FFmpeg 확인
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode == 0:
            logger.info("  ✅ FFmpeg 확인")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    logger.error("  ❌ FFmpeg 없음")
    logger.info("  해결: https://ffmpeg.org/download.html")
    return False

def check_step8_rendering():
    """8단계: 최종 렌더링"""
    # FFmpeg 확인 (이미 위에서 확인했지만)
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode == 0:
            logger.info("  ✅ FFmpeg 확인")
            return True
    except:
        pass
    
    logger.error("  ❌ FFmpeg 없음")
    return False

def main():
    """메인 함수"""
    logger.info("=" * 60)
    logger.info("8단계 파이프라인 검증")
    logger.info("=" * 60)
    
    steps = [
        (1, "영어 TTS 생성", check_step1_english_tts),
        (2, "영어 비디오 생성", check_step2_video_generation),
        (3, "STT 처리", check_step3_stt),
        (4, "번역", check_step4_translation),
        (5, "한국어 TTS 생성", check_step5_korean_tts),
        (6, "자막 생성", check_step6_subtitles),
        (7, "비디오 음소거", check_step7_video_mute),
        (8, "최종 렌더링", check_step8_rendering),
    ]
    
    results = []
    for step_num, step_name, check_func in steps:
        result = validate_step(step_num, step_name, check_func)
        results.append((step_num, step_name, result))
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("검증 결과 요약")
    logger.info("=" * 60)
    
    passed = sum(1 for _, _, r in results if r)
    failed = sum(1 for _, _, r in results if not r)
    
    logger.info(f"✅ 통과: {passed}/8")
    logger.info(f"❌ 실패: {failed}/8")
    logger.info("")
    
    if failed > 0:
        logger.info("실패한 단계:")
        for step_num, step_name, result in results:
            if not result:
                logger.info(f"  [{step_num}/8] {step_name}")
        logger.info("")
        logger.info("해결 방법:")
        logger.info("  python run_with_auto_fix.py 실행")
        return False
    else:
        logger.info("✅ 모든 단계 준비 완료!")
        logger.info("")
        logger.info("비디오 생성 준비가 완료되었습니다.")
        logger.info("  실행: python main.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
