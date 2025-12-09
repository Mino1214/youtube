@echo off
REM 호환 버전으로 단계별 의존성 설치 (충돌 방지)

echo ============================================================
echo 호환 버전 의존성 설치 (충돌 방지)
echo ============================================================
echo.

REM 가상환경 확인
if not exist venv (
    echo ❌ 가상환경이 없습니다.
    echo 먼저 가상환경을 생성하세요: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/6] pip 업그레이드...
python -m pip install --upgrade pip setuptools wheel
echo.

echo [2/6] PyTorch GPU 버전 설치 (먼저 설치 - NumPy 자동 설치됨)...
echo CUDA 13.1 호환 버전 설치 시도...
echo (CUDA 12.4 버전 사용 - CUDA 13.1과 호환)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo ⚠️  CUDA 12.4 버전 실패, CUDA 12.1로 시도...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo ⚠️  GPU 버전 실패, CPU 버전으로 시도...
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    )
)
echo.

echo [3/6] NumPy 확인 및 설치...
python -c "import numpy; print('NumPy 버전:', numpy.__version__)" 2>nul
if errorlevel 1 (
    echo NumPy 설치 중...
    pip install "numpy>=1.24.0,<2.0.0"
) else (
    echo ✅ NumPy 이미 설치됨
)
echo.

echo [4/6] Transformers & Diffusers 설치 (호환 버전)...
pip install "transformers>=4.40.0,<5.0.0"
pip install "diffusers>=0.27.0,<0.30.0"
pip install "huggingface-hub>=0.20.0,<1.0.0"
echo.

echo [5/6] 기타 ML 라이브러리 설치...
pip install "accelerate>=0.24.0,<1.0.0"
pip install "sentencepiece>=0.1.99"
pip install "protobuf>=3.20.0,<6.0.0"
echo.

echo [6/6] 나머지 패키지 설치...
pip install "openai-whisper>=20231117"
pip install "librosa>=0.10.0,<1.0.0"
pip install "piper-tts>=1.2.0"
pip install "ffmpeg-python>=0.2.0"
pip install "moviepy>=1.0.3"
pip install "yt-dlp>=2023.11.16"
pip install "pyyaml>=6.0.1"
pip install "tqdm>=4.66.0"
pip install "requests>=2.31.0"
pip install "Pillow>=10.0.0"
pip install "imageio>=2.31.0"
pip install "imageio-ffmpeg>=0.4.9"
pip install "safetensors>=0.4.0,<1.0.0"
pip install "invisible-watermark>=0.2.0,<1.0.0"
echo.

echo ============================================================
echo 설치 확인 중...
echo ============================================================
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import numpy; print('✅ NumPy:', numpy.__version__)"
python -c "import transformers; print('✅ Transformers:', transformers.__version__)"
python -c "import diffusers; print('✅ Diffusers:', diffusers.__version__)"
python -c "import whisper; print('✅ Whisper: OK')" 2>nul || echo "⚠️  Whisper 확인 필요"

echo.
echo ============================================================
echo ✅ 의존성 설치 완료!
echo ============================================================
echo.
pause
