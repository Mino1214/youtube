@echo off
REM 통합 진단 및 수리 스크립트 (원클릭)

echo ============================================================
echo AI Video Generator - 통합 진단 및 수리
echo ============================================================
echo.

REM 가상환경 확인
if not exist venv (
    echo [진단] 가상환경 없음
    echo [수리] 가상환경 생성 중...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 가상환경 생성 실패
        pause
        exit /b 1
    )
    echo ✅ 가상환경 생성 완료
)

call venv\Scripts\activate.bat

echo.
echo ============================================================
echo 1단계: Python 환경 확인
echo ============================================================
python --version
if errorlevel 1 (
    echo ❌ Python을 찾을 수 없습니다.
    pause
    exit /b 1
)
echo ✅ Python 확인 완료
echo.

echo ============================================================
echo 2단계: 필수 패키지 확인
echo ============================================================
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul || (
    echo ❌ PyTorch 없음
    echo [수리] PyTorch 설치 중...
    call install_pytorch_cuda131.bat
)

python -c "import numpy; print('✅ NumPy:', numpy.__version__)" 2>nul || (
    echo ❌ NumPy 없음
    echo [수리] NumPy 설치 중...
    pip install "numpy>=1.24.0,<2.0.0"
)

python -c "import transformers; print('✅ Transformers:', transformers.__version__)" 2>nul || (
    echo ❌ Transformers 없음
    echo [수리] Transformers 설치 중...
    pip install "transformers>=4.40.0,<5.0.0"
)

python -c "import diffusers; print('✅ Diffusers:', diffusers.__version__)" 2>nul || (
    echo ❌ Diffusers 없음
    echo [수리] Diffusers 설치 중...
    pip install "diffusers>=0.27.0,<0.30.0"
)

python -c "import whisper; print('✅ Whisper: OK')" 2>nul || (
    echo ❌ Whisper 없음
    echo [수리] Whisper 설치 중...
    pip install "openai-whisper>=20231117"
)

python -c "from piper import PiperVoice; print('✅ Piper TTS: OK')" 2>nul || (
    echo ❌ Piper TTS 없음
    echo [수리] Piper TTS 설치 중...
    pip install "piper-tts>=1.2.0"
)
echo.

echo ============================================================
echo 3단계: GPU 확인
echo ============================================================
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA 사용 가능')
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Compute Capability: {torch.cuda.get_device_capability(0)}')
    print(f'   CUDA 버전: {torch.version.cuda}')
    # GPU 테스트
    try:
        test = torch.zeros(1).cuda()
        del test
        print('   ✅ GPU 연산 테스트 성공')
    except RuntimeError as e:
        if 'kernel image' in str(e).lower() or 'no kernel' in str(e).lower():
            print('   ❌ GPU Compute Capability 호환성 문제')
            print('   [수리 필요] PyTorch Nightly 버전 설치 필요')
            exit(1)
        else:
            raise
else:
    print('❌ CUDA를 사용할 수 없습니다.')
    print('   [수리 필요] PyTorch GPU 버전 설치 필요')
    exit(1)
"
if errorlevel 1 (
    echo.
    echo [수리] PyTorch Nightly 버전 설치 중...
    call install_pytorch_nightly.bat
)
echo.

echo ============================================================
echo 4단계: 모델 확인
echo ============================================================
python -c "
from pathlib import Path
import sys

print('필수 모델 확인 중...')
print()

# 영어 TTS 모델
voice_name = 'en_US-amy-medium'
hf_cache = Path.home() / '.cache' / 'huggingface' / 'hub'
hf_model_dir = hf_cache / 'models--rhasspy--piper-voices'
found_tts = False

if hf_model_dir.exists():
    snapshots_dir = hf_model_dir / 'snapshots'
    if snapshots_dir.exists():
        for snapshot in snapshots_dir.iterdir():
            for onnx_file in snapshot.rglob(f'{voice_name}.onnx'):
                print(f'✅ 영어 TTS 모델 발견: {onnx_file}')
                found_tts = True
                break
            if found_tts:
                break

if not found_tts:
    save_dir = Path.home() / '.local' / 'share' / 'piper' / 'voices' / 'en' / 'en_US' / 'amy' / 'medium'
    if (save_dir / f'{voice_name}.onnx').exists():
        print(f'✅ 영어 TTS 모델 발견: {save_dir}')
        found_tts = True

if not found_tts:
    print('❌ 영어 TTS 모델 없음')
    print('   [수리 필요] 모델 다운로드 필요')
    sys.exit(1)

# SDXL 모델
sdxl_dir = hf_cache / 'models--stabilityai--stable-diffusion-xl-base-1.0'
if sdxl_dir.exists():
    print('✅ Stable Diffusion XL 모델 발견')
else:
    print('⚠️  Stable Diffusion XL 모델 없음 (첫 실행 시 자동 다운로드)')

# SVD 모델
svd_dir = hf_cache / 'models--stabilityai--stable-video-diffusion-img2vid'
if svd_dir.exists():
    print('✅ Stable Video Diffusion 모델 발견')
else:
    print('⚠️  Stable Video Diffusion 모델 없음 (첫 실행 시 자동 다운로드)')
"
if errorlevel 1 (
    echo.
    echo [수리] 모델 다운로드 중...
    python download_all_models.py --auto
)
echo.

echo ============================================================
echo 5단계: 입력 파일 확인
echo ============================================================
if exist input.txt (
    echo ✅ input.txt 발견
) else if exist input_text.txt (
    echo ✅ input_text.txt 발견
) else (
    echo ❌ 입력 파일 없음
    echo [수리] input.txt 생성 중...
    echo Welcome to AI video generation. > input.txt
    echo ✅ input.txt 생성 완료
)
echo.

echo ============================================================
echo 6단계: FFmpeg 확인
echo ============================================================
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ❌ FFmpeg 없음
    echo ⚠️  FFmpeg를 설치하세요: https://ffmpeg.org/download.html
) else (
    echo ✅ FFmpeg 확인 완료
)
echo.

echo ============================================================
echo 진단 완료!
echo ============================================================
echo.
echo 모든 문제를 수리했습니다.
echo.
echo 비디오 생성을 시작하시겠습니까? (Y/N)
set /p choice=
if /i "%choice%"=="Y" (
    echo.
    echo ============================================================
    echo 비디오 생성 시작
    echo ============================================================
    python main.py
) else (
    echo 취소되었습니다.
)

echo.
pause
