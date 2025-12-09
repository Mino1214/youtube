#!/bin/bash
# Python 3.10 가상환경 설정 스크립트 (Linux/Mac)

echo "============================================================"
echo "Python 3.10 가상환경 설정"
echo "============================================================"
echo ""

# Python 3.10 확인
echo "[1/5] Python 3.10 확인 중..."
if ! command -v python3.10 &> /dev/null; then
    echo ""
    echo "❌ Python 3.10을 찾을 수 없습니다."
    echo "Python 3.10을 설치하거나 PATH에 추가해주세요."
    exit 1
fi
python3.10 --version
echo "✅ Python 3.10 확인 완료"
echo ""

# 기존 가상환경 삭제 (선택사항)
if [ -d "venv" ]; then
    echo "[2/5] 기존 가상환경 삭제 중..."
    rm -rf venv
    echo "✅ 기존 가상환경 삭제 완료"
    echo ""
else
    echo "[2/5] 기존 가상환경 없음"
    echo ""
fi

# 가상환경 생성
echo "[3/5] Python 3.10 가상환경 생성 중..."
python3.10 -m venv venv
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 가상환경 생성 실패"
    exit 1
fi
echo "✅ 가상환경 생성 완료"
echo ""

# 가상환경 활성화 및 pip 업그레이드
echo "[4/5] pip 업그레이드 중..."
source venv/bin/activate
pip install --upgrade pip
echo "✅ pip 업그레이드 완료"
echo ""

# 필수 패키지 설치
echo "[5/5] 필수 패키지 설치 중..."
echo ""
echo "NumPy 설치 중..."
pip install numpy==1.24.0
echo ""
echo "PyTorch 설치 중..."
pip install torch
echo ""
echo "기타 필수 패키지 설치 중..."
pip install -r requirements.txt
echo ""

echo "============================================================"
echo "✅ 가상환경 설정 완료!"
echo "============================================================"
echo ""
echo "다음 명령어로 가상환경을 활성화하세요:"
echo "  source venv/bin/activate"
echo ""
