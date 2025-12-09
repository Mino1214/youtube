"""
원클릭 실행 스크립트 - 자동 진단 및 수리 후 실행
8단계 파이프라인을 순차적으로 검증하고 문제를 수리합니다.
"""
import sys
import subprocess
import importlib
from pathlib import Path
import logging
import yaml
import os

# torch 모듈 재로드를 위해
if 'torch' in sys.modules:
    del sys.modules['torch']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineDiagnostic:
    """파이프라인 진단 및 수리 클래스"""
    
    def __init__(self):
        self.issues = []
        self.fixed = []
        
    def check_step(self, step_name, check_func, fix_func=None):
        """단계별 확인 및 수리"""
        logger.info(f"\n{'='*60}")
        logger.info(f"{step_name} 확인 중...")
        logger.info(f"{'='*60}")
        
        try:
            result = check_func()
            if result:
                logger.info(f"✅ {step_name} 정상")
                return True
            else:
                logger.warning(f"⚠️  {step_name} 문제 발견")
                self.issues.append(step_name)
                
                if fix_func:
                    logger.info(f"[수리] {step_name} 수리 중...")
                    if fix_func():
                        logger.info(f"✅ {step_name} 수리 완료")
                        self.fixed.append(step_name)
                        return True
                    else:
                        logger.error(f"❌ {step_name} 수리 실패")
                        return False
                else:
                    return False
        except Exception as e:
            logger.error(f"❌ {step_name} 확인 중 오류: {e}")
            self.issues.append(step_name)
            return False
    
    def check_python_env(self):
        """1. Python 환경 확인"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(f"Python {version.major}.{version.minor}는 지원하지 않습니다. Python 3.8 이상 필요.")
            return False
        logger.info(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_venv(self):
        """가상환경 확인"""
        venv_path = Path("venv")
        if not venv_path.exists():
            logger.warning("가상환경이 없습니다.")
            return False
        return True
    
    def check_packages(self):
        """2. 필수 패키지 확인"""
        packages = {
            "torch": ("torch", "torch"),
            "numpy": ("numpy", "numpy>=1.24.0,<2.0.0"),
            "transformers": ("transformers", "transformers>=4.40.0,<5.0.0"),
            "diffusers": ("diffusers", "diffusers>=0.27.0,<0.30.0"),
            "whisper": ("whisper", "openai-whisper>=20231117"),
            "piper": ("piper", "piper-tts>=1.2.0"),
            "yaml": ("yaml", "pyyaml>=6.0.1"),
        }
        
        missing = []
        for name, (import_name, install_cmd) in packages.items():
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                logger.info(f"  ✅ {name}: {version}")
            except ImportError:
                logger.warning(f"  ❌ {name} 없음")
                missing.append((name, install_cmd))
        
        if missing:
            logger.info(f"[수리] {len(missing)}개 패키지 설치 중...")
            for name, install_cmd in missing:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install"] + install_cmd.split(),
                                 check=True, capture_output=True, timeout=300)
                    logger.info(f"  ✅ {name} 설치 완료")
                except subprocess.TimeoutExpired:
                    logger.error(f"  ❌ {name} 설치 시간 초과")
                    return False
                except subprocess.CalledProcessError as e:
                    logger.error(f"  ❌ {name} 설치 실패: {e}")
                    return False
        
        return True
    
    def check_gpu(self):
        """3. GPU 확인"""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.error("CUDA를 사용할 수 없습니다.")
                return False
            
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            cuda_version = torch.version.cuda
            
            logger.info(f"  GPU: {gpu_name}")
            logger.info(f"  Compute Capability: {capability[0]}.{capability[1]}")
            logger.info(f"  CUDA: {cuda_version}")
            
            # GPU 테스트
            try:
                test = torch.zeros(100, 100).cuda()
                result = torch.matmul(test, test)
                del test, result
                torch.cuda.empty_cache()
                logger.info("  ✅ GPU 연산 테스트 성공")
                return True
            except RuntimeError as e:
                error_str = str(e).lower()
                if "kernel image" in error_str or "no kernel" in error_str:
                    logger.error("  ❌ GPU Compute Capability 호환성 문제")
                    logger.info("  [수리 필요] PyTorch Nightly 버전 설치")
                    return False
                else:
                    raise
        except ImportError:
            logger.error("PyTorch가 설치되지 않았습니다.")
            return False
    
    def check_models(self):
        """4. 모델 확인"""
        issues = []
        
        # 영어 TTS 모델
        voice_name = "en_US-amy-medium"
        found = False
        
        # HuggingFace 캐시 확인
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        hf_model_dir = hf_cache / "models--rhasspy--piper-voices"
        
        if hf_model_dir.exists():
            snapshots_dir = hf_model_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    for onnx_file in snapshot.rglob(f"{voice_name}.onnx"):
                        logger.info(f"  ✅ 영어 TTS 모델: {onnx_file}")
                        found = True
                        break
                    if found:
                        break
        
        # download_all_models.py 저장 경로 확인
        if not found:
            save_dir = Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium"
            if (save_dir / f"{voice_name}.onnx").exists():
                logger.info(f"  ✅ 영어 TTS 모델: {save_dir}")
                found = True
        
        if not found:
            logger.warning("  ❌ 영어 TTS 모델 없음")
            issues.append("tts_model")
        
        # SDXL 모델 (선택사항 - 첫 실행 시 자동 다운로드)
        sdxl_dir = hf_cache / "models--stabilityai--stable-diffusion-xl-base-1.0"
        if sdxl_dir.exists():
            logger.info("  ✅ Stable Diffusion XL 모델 발견")
        else:
            logger.info("  ⚠️  Stable Diffusion XL 모델 없음 (첫 실행 시 자동 다운로드)")
        
        return len(issues) == 0
    
    def check_input_file(self):
        """5. 입력 파일 확인"""
        for filename in ["input.txt", "input_text.txt"]:
            if Path(filename).exists():
                content = Path(filename).read_text(encoding="utf-8").strip()
                if content:
                    logger.info(f"  ✅ {filename} 발견 ({len(content)}자)")
                    return True
                else:
                    logger.warning(f"  ⚠️  {filename}가 비어있습니다.")
        
        logger.warning("  ❌ 입력 파일 없음")
        return False
    
    def check_config(self):
        """6. 설정 파일 확인"""
        config_path = Path("config.yaml")
        if not config_path.exists():
            logger.warning("  ❌ config.yaml 없음")
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 필수 설정 확인
            required_sections = ["llm", "tts", "whisper", "video_generation"]
            missing = [s for s in required_sections if s not in config]
            
            if missing:
                logger.warning(f"  ⚠️  설정 섹션 누락: {', '.join(missing)}")
            else:
                logger.info("  ✅ config.yaml 정상")
            
            return True
        except Exception as e:
            logger.error(f"  ❌ config.yaml 읽기 실패: {e}")
            return False
    
    def check_ffmpeg(self):
        """7. FFmpeg 확인"""
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  capture_output=True, 
                                  timeout=5)
            if result.returncode == 0:
                logger.info("  ✅ FFmpeg 확인 완료")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        logger.warning("  ❌ FFmpeg 없음")
        return False
    
    def fix_tts_model(self):
        """TTS 모델 다운로드"""
        try:
            logger.info("  모델 다운로드 중...")
            result = subprocess.run([sys.executable, "download_all_models.py", "--auto"],
                                   timeout=3600, capture_output=True, text=True)
            if result.returncode == 0:
                return True
            else:
                logger.error(f"  다운로드 실패: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("  다운로드 시간 초과")
            return False
        except Exception as e:
            logger.error(f"  다운로드 오류: {e}")
            return False
    
    def fix_input_file(self):
        """입력 파일 생성"""
        try:
            Path("input.txt").write_text(
                "Welcome to AI video generation. This is a test video.\n",
                encoding="utf-8"
            )
            return True
        except Exception as e:
            logger.error(f"  입력 파일 생성 실패: {e}")
            return False
    
    def fix_gpu(self):
        """GPU 문제 수리 (PyTorch Nightly 설치)"""
        logger.info("  PyTorch Nightly 설치 중...")
        logger.info("  이 작업은 몇 분이 걸릴 수 있습니다...")
        
        # 기존 PyTorch 제거
        try:
            logger.info("  [1/3] 기존 PyTorch 제거 중...")
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                          "torch", "torchvision", "torchaudio"],
                         capture_output=True, timeout=60)
        except Exception as e:
            logger.warning(f"  기존 PyTorch 제거 중 오류 (무시): {e}")
        
        # CUDA 버전별로 시도
        cuda_versions = [
            ("cu124", "CUDA 12.4"),
            ("cu121", "CUDA 12.1"),
            ("cu118", "CUDA 11.8"),
        ]
        
        for cuda_version, cuda_name in cuda_versions:
            try:
                logger.info(f"  [2/3] PyTorch Nightly 설치 시도 ({cuda_name})...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--pre",
                     "torch", "torchvision", "torchaudio",
                     "--index-url", f"https://download.pytorch.org/whl/nightly/{cuda_version}"],
                    timeout=900,  # 15분
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"  ✅ PyTorch Nightly 설치 완료 ({cuda_name})")
                    
                    # 설치 확인
                    logger.info("  [3/3] GPU 테스트 중...")
                    try:
                        import torch
                        # 모듈 다시 로드
                        importlib.reload(torch)
                        
                        if torch.cuda.is_available():
                            # GPU 테스트
                            test = torch.zeros(1).cuda()
                            result_test = test * 2
                            del test, result_test
                            torch.cuda.empty_cache()
                            
                            gpu_name = torch.cuda.get_device_name(0)
                            capability = torch.cuda.get_device_capability(0)
                            logger.info(f"  ✅ GPU 테스트 성공!")
                            logger.info(f"     GPU: {gpu_name}")
                            logger.info(f"     Compute Capability: {capability[0]}.{capability[1]}")
                            return True
                        else:
                            logger.error("  ❌ CUDA를 사용할 수 없습니다.")
                            continue
                    except RuntimeError as e:
                        error_str = str(e).lower()
                        if "kernel image" in error_str or "no kernel" in error_str:
                            logger.warning(f"  ⚠️  {cuda_name} 버전도 호환성 문제")
                            logger.info(f"  다음 CUDA 버전으로 시도...")
                            continue
                        else:
                            raise
                else:
                    logger.warning(f"  ⚠️  {cuda_name} 설치 실패")
                    if result.stderr:
                        logger.debug(f"  오류: {result.stderr[:500]}")
                    continue
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"  ⚠️  {cuda_name} 설치 시간 초과")
                continue
            except Exception as e:
                logger.warning(f"  ⚠️  {cuda_name} 설치 중 오류: {e}")
                continue
        
        logger.error("  ❌ 모든 CUDA 버전 설치 실패")
        logger.info("  수동 해결 방법:")
        logger.info("    1. install_pytorch_nightly.bat 실행")
        logger.info("    2. 또는 PyTorch 소스에서 빌드")
        return False

def diagnose_and_fix():
    """전체 진단 및 수리"""
    logger.info("=" * 60)
    logger.info("AI Video Generator - 통합 진단 및 수리")
    logger.info("=" * 60)
    logger.info("")
    
    diag = PipelineDiagnostic()
    
    # 1. Python 환경
    diag.check_step("Python 환경", diag.check_python_env)
    
    # 2. 가상환경
    if not diag.check_venv():
        logger.warning("가상환경이 없습니다. 시스템 Python을 사용합니다.")
    
    # 3. 필수 패키지
    diag.check_step("필수 패키지", diag.check_packages)
    
    # 4. GPU - 먼저 상세 진단
    logger.info("")
    logger.info("=" * 60)
    logger.info("GPU 상세 진단 중...")
    logger.info("=" * 60)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            pytorch_version = torch.__version__
            cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "알 수 없음"
            
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"Compute Capability: {capability[0]}.{capability[1]}")
            logger.info(f"PyTorch 버전: {pytorch_version}")
            logger.info(f"CUDA 버전 (빌드): {cuda_version}")
            logger.info("")
            
            # 호환성 분석
            is_nightly = 'dev' in pytorch_version or 'nightly' in pytorch_version.lower()
            major, minor = capability
            
            logger.info("호환성 분석:")
            if major >= 9:
                if not is_nightly:
                    logger.warning("  ⚠️  Compute Capability 9.0+는 PyTorch Nightly가 필요합니다.")
            elif major == 8 and minor >= 9:
                if not is_nightly:
                    logger.warning("  ⚠️  최신 Ada GPU는 PyTorch Nightly가 권장됩니다.")
            elif major < 7:
                logger.error("  ❌ Compute Capability 7.0 미만은 PyTorch가 지원하지 않을 수 있습니다.")
            
            logger.info("")
    except Exception as e:
        logger.warning(f"GPU 정보 확인 중 오류: {e}")
    
    # GPU 테스트
    gpu_ok = diag.check_step("GPU", diag.check_gpu, None)
    
    if not gpu_ok:
        logger.info("")
        logger.info("=" * 60)
        logger.info("GPU 문제 발견 - 상세 진단 권장")
        logger.info("=" * 60)
        logger.info("")
        logger.info("정확한 문제 파악을 위해 상세 진단을 실행하세요:")
        logger.info("  DIAGNOSE_GPU.bat 실행")
        logger.info("")
        logger.info("또는 자동 수리 시도:")
        logger.info("  fix_gpu_auto.bat 실행")
        logger.info("")
        
        # 사용자 선택
        response = input("지금 자동 수리를 시도하시겠습니까? (Y/N): ")
        if response.upper() == "Y":
            logger.info("")
            logger.info("GPU 수리 중...")
            if diag.fix_gpu():
                # 모듈 재로드 후 다시 확인
                if 'torch' in sys.modules:
                    del sys.modules['torch']
                gpu_ok = diag.check_step("GPU (재확인)", diag.check_gpu, None)
            
            if not gpu_ok:
                logger.error("")
                logger.error("=" * 60)
                logger.error("❌ GPU 문제를 해결할 수 없습니다.")
                logger.error("=" * 60)
                logger.error("")
                logger.error("다음 단계:")
                logger.error("  1. DIAGNOSE_GPU.bat 실행하여 상세 진단")
                logger.error("  2. 진단 결과에 따라 수동 해결")
                logger.error("")
                return False
        else:
            logger.info("")
            logger.info("상세 진단 후 다시 시도하세요.")
            logger.info("  DIAGNOSE_GPU.bat 실행")
            return False
    
    # 5. 모델
    diag.check_step("모델", diag.check_models, diag.fix_tts_model)
    
    # 6. 입력 파일
    diag.check_step("입력 파일", diag.check_input_file, diag.fix_input_file)
    
    # 7. 설정 파일
    diag.check_step("설정 파일", diag.check_config)
    
    # 8. FFmpeg
    diag.check_step("FFmpeg", diag.check_ffmpeg)
    
    # 결과 요약
    logger.info("")
    logger.info("=" * 60)
    logger.info("진단 결과")
    logger.info("=" * 60)
    logger.info(f"발견된 문제: {len(diag.issues)}개")
    if diag.issues:
        for issue in diag.issues:
            logger.info(f"  - {issue}")
    logger.info(f"수리 완료: {len(diag.fixed)}개")
    if diag.fixed:
        for fix in diag.fixed:
            logger.info(f"  ✅ {fix}")
    logger.info("=" * 60)
    logger.info("")
    
    if diag.issues:
        logger.warning("일부 문제가 남아있습니다. 수동으로 해결하세요.")
        return False
    
    return True

def main():
    """메인 함수"""
    # 진단 및 수리
    if not diagnose_and_fix():
        logger.error("진단 실패. 문제를 해결한 후 다시 시도하세요.")
        return False
    
    logger.info("=" * 60)
    logger.info("✅ 모든 준비 완료!")
    logger.info("=" * 60)
    logger.info("")
    
    # 실행 여부 확인
    response = input("비디오 생성을 시작하시겠습니까? (Y/N): ")
    if response.upper() != "Y":
        logger.info("취소되었습니다.")
        return False
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("비디오 생성 시작")
    logger.info("=" * 60)
    logger.info("")
    
    # main.py 실행
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ 비디오 생성 완료!")
        logger.info("=" * 60)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"비디오 생성 실패: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("\n중단되었습니다.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        sys.exit(1)
