#!/bin/bash
# PyTorch 업그레이드 스크립트 (Ubuntu/Linux)

echo "========================================"
echo "PyTorch 업그레이드 스크립트"
echo "RTX 5060 Ti 호환성 문제 해결"
echo "========================================"
echo ""

echo "[1/3] 기존 PyTorch 제거 중..."
pip uninstall torch torchvision torchaudio -y

echo ""
echo "[2/3] 최신 PyTorch 설치 중 (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "[3/3] 설치 확인 중..."
python check_cuda.py

echo ""
echo "========================================"
echo "완료! check_cuda.py를 실행하여 확인하세요."
echo "========================================"

