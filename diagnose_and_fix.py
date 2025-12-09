"""
통합 진단 및 수리 스크립트 (Python 버전)
원클릭으로 모든 문제를 진단하고 수리한 후 실행
"""
import sys
import subprocess
import importlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_python():
    """Python 버전 확인"""
    logger.info("=" * 60)
    logger.info("1단계: Python 환경 확인")
    logger.info("=" * 60)
    version = sys.version_info
    logger.info(f"Python 버전: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8 이상이 필요합니다.")
        return False
    logger.info("✅ Python 버전 확인 완료\n")
    return True

def check_and_install_package(package_name, import_name=None, install_cmd=None):
    """패키지 확인 및 설치"""
    if import_name is None:
        import_name = package_name
    if install_cmd is None:
        install_cmd = f"pip install {package_name}"
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        logger.info(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        logger.warning(f"❌ {package_name} 없음")
        logger.info(f"[수리] {package_name} 설치 중...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + install_cmd.split()[1:], 
                         check=True, capture_output=True)
            logger.info(f"✅ {package_name} 설치 완료")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {package_name} 설치 실패: {e}")
            return False

def check_gpu():
    """GPU 확인"""
    logger.info("=" * 60)
    logger.info("2단계: GPU 확인")
    logger.info("=" * 60)
    
    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("❌ CUDA를 사용할 수 없습니다.")
            logger.info("[수리] PyTorch GPU 버전 설치 필요")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        cuda_version = torch.version.cuda
        
        logger.info(f"✅ CUDA 사용 가능")
        logger.info(f"   GPU: {gpu_name}")
        logger.info(f"   Compute Capability: {capability[0]}.{capability[1]}")
        logger.info(f"   CUDA 버전: {cuda_version}")
        
        # GPU 테스트
        try:
            test = torch.zeros(1).cuda()
            result = test * 2
            del test, result
            torch.cuda.empty_cache()
            logger.info("   ✅ GPU 연산 테스트 성공\n")
            return True
        except RuntimeError as e:
            error_str = str(e).lower()
            if "kernel image" in error_str or "no kernel" in error_str:
                logger.error("   ❌ GPU Compute Capability 호환성 문제")
                logger.info("   [수리 필요] PyTorch Nightly 버전 설치 필요\n")
                return False
            else:
                raise
    except ImportError:
        logger.error("❌ PyTorch가 설치되지 않았습니다.")
        logger.info("[수리] PyTorch 설치 필요\n")
        return False

def check_models():
    """모델 확인"""
    logger.info("=" * 60)
    logger.info("3단계: 모델 확인")
    logger.info("=" * 60)
    
    issues = []
    
    # 영어 TTS 모델
    voice_name = "en_US-amy-medium"
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    hf_model_dir = hf_cache / "models--rhasspy--piper-voices"
    found_tts = False
    
    if hf_model_dir.exists():
        snapshots_dir = hf_model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                for onnx_file in snapshot.rglob(f"{voice_name}.onnx"):
                    logger.info(f"✅ 영어 TTS 모델 발견: {onnx_file}")
                    found_tts = True
                    break
                if found_tts:
                    break
    
    if not found_tts:
        save_dir = Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium"
        if (save_dir / f"{voice_name}.onnx").exists():
            logger.info(f"✅ 영어 TTS 모델 발견: {save_dir}")
            found_tts = True
    
    if not found_tts:
        logger.warning("❌ 영어 TTS 모델 없음")
        issues.append("tts")
    
    # SDXL 모델
    sdxl_dir = hf_cache / "models--stabilityai--stable-diffusion-xl-base-1.0"
    if sdxl_dir.exists():
        logger.info("✅ Stable Diffusion XL 모델 발견")
    else:
        logger.info("⚠️  Stable Diffusion XL 모델 없음 (첫 실행 시 자동 다운로드)")
    
    # SVD 모델
    svd_dir = hf_cache / "models--stabilityai--stable-video-diffusion-img2vid"
    if svd_dir.exists():
        logger.info("✅ Stable Video Diffusion 모델 발견")
    else:
        logger.info("⚠️  Stable Video Diffusion 모델 없음 (첫 실행 시 자동 다운로드)")
    
    logger.info("")
    return issues

def check_input_file():
    """입력 파일 확인"""
    logger.info("=" * 60)
    logger.info("4단계: 입력 파일 확인")
    logger.info("=" * 60)
    
    for filename in ["input.txt", "input_text.txt"]:
        if Path(filename).exists():
            logger.info(f"✅ {filename} 발견\n")
            return True
    
    logger.warning("❌ 입력 파일 없음")
    logger.info("[수리] input.txt 생성 중...")
    Path("input.txt").write_text("Welcome to AI video generation.\n", encoding="utf-8")
    logger.info("✅ input.txt 생성 완료\n")
    return True

def check_ffmpeg():
    """FFmpeg 확인"""
    logger.info("=" * 60)
    logger.info("5단계: FFmpeg 확인")
    logger.info("=" * 60)
    
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, 
                              timeout=5)
        if result.returncode == 0:
            logger.info("✅ FFmpeg 확인 완료\n")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    logger.warning("❌ FFmpeg 없음")
    logger.info("⚠️  FFmpeg를 설치하세요: https://ffmpeg.org/download.html\n")
    return False

def fix_issues(issues):
    """문제 수리"""
    if not issues:
        return True
    
    logger.info("=" * 60)
    logger.info("문제 수리 중...")
    logger.info("=" * 60)
    
    if "gpu" in issues:
        logger.info("PyTorch GPU 버전 설치 중...")
        # install_pytorch_cuda131.bat 호출 또는 직접 설치
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", 
                          "torch", "torchvision", "torchaudio",
                          "--index-url", "https://download.pytorch.org/whl/cu124"],
                         check=True)
        except:
            logger.error("PyTorch GPU 설치 실패. 수동으로 install_pytorch_cuda131.bat 실행하세요.")
            return False
    
    if "tts" in issues:
        logger.info("영어 TTS 모델 다운로드 중...")
        try:
            subprocess.run([sys.executable, "download_all_models.py", "--auto"],
                         check=True, timeout=3600)
        except:
            logger.error("모델 다운로드 실패. 수동으로 download_all_models.py --auto 실행하세요.")
            return False
    
    logger.info("✅ 문제 수리 완료\n")
    return True

def main():
    """메인 함수"""
    logger.info("=" * 60)
    logger.info("AI Video Generator - 통합 진단 및 수리")
    logger.info("=" * 60)
    logger.info("")
    
    # 1. Python 확인
    if not check_python():
        return False
    
    # 2. 필수 패키지 확인
    logger.info("=" * 60)
    logger.info("필수 패키지 확인")
    logger.info("=" * 60)
    
    packages = [
        ("torch", "torch", "torch"),
        ("numpy", "numpy", "numpy>=1.24.0,<2.0.0"),
        ("transformers", "transformers", "transformers>=4.40.0,<5.0.0"),
        ("diffusers", "diffusers", "diffusers>=0.27.0,<0.30.0"),
        ("whisper", "whisper", "openai-whisper>=20231117"),
        ("piper", "piper", "piper-tts>=1.2.0"),
    ]
    
    issues = []
    for pkg_name, import_name, install_cmd in packages:
        if not check_and_install_package(pkg_name, import_name, install_cmd):
            issues.append(pkg_name)
    
    logger.info("")
    
    # 3. GPU 확인
    gpu_ok = check_gpu()
    if not gpu_ok:
        issues.append("gpu")
    
    # 4. 모델 확인
    model_issues = check_models()
    issues.extend(model_issues)
    
    # 5. 입력 파일 확인
    check_input_file()
    
    # 6. FFmpeg 확인
    check_ffmpeg()
    
    # 문제 수리
    if issues:
        if not fix_issues(issues):
            logger.error("일부 문제를 수리하지 못했습니다.")
            return False
    
    logger.info("=" * 60)
    logger.info("✅ 진단 완료! 모든 준비가 완료되었습니다.")
    logger.info("=" * 60)
    logger.info("")
    
    # 실행 여부 확인
    response = input("비디오 생성을 시작하시겠습니까? (Y/N): ")
    if response.upper() == "Y":
        logger.info("")
        logger.info("=" * 60)
        logger.info("비디오 생성 시작")
        logger.info("=" * 60)
        logger.info("")
        try:
            subprocess.run([sys.executable, "main.py"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"비디오 생성 실패: {e}")
            return False
    else:
        logger.info("취소되었습니다.")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n중단되었습니다.")
        sys.exit(1)
