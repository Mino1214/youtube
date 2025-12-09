@echo off
REM 모델 위치 확인 스크립트

echo ============================================================
echo 모델 위치 확인
echo ============================================================
echo.

if not exist venv (
    echo ❌ 가상환경이 없습니다.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

python -c "
from pathlib import Path
import sys

print('=' * 60)
print('영어 TTS 모델 (en_US-amy-medium) 검색 중...')
print('=' * 60)
print()

voice_name = 'en_US-amy-medium'

# HuggingFace 캐시 확인
hf_cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
hf_model_dir = hf_cache_dir / 'models--rhasspy--piper-voices'
hf_paths = []
if hf_model_dir.exists():
    snapshots_dir = hf_model_dir / 'snapshots'
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
            hf_paths.append(latest_snapshot / 'en' / 'en_US' / 'amy' / 'medium' / f'{voice_name}.onnx')
            hf_paths.append(latest_snapshot / 'en' / 'en_US' / 'amy' / 'medium' / 'model.onnx')
            print(f'✅ HuggingFace 캐시 발견: {hf_model_dir}')
            print(f'   최신 스냅샷: {latest_snapshot.name}')

possible_paths = hf_paths + [
    Path.home() / '.local' / 'share' / 'piper' / 'voices' / 'en' / 'en_US' / 'amy' / 'medium' / f'{voice_name}.onnx',
    Path.home() / '.local' / 'share' / 'piper' / 'voices' / 'en' / 'en_US' / 'amy' / 'medium' / 'model.onnx',
    Path.home() / '.local' / 'share' / 'piper' / 'voices' / 'en' / 'en_US' / f'{voice_name}.onnx',
    Path('models') / 'tts' / f'{voice_name}.onnx',
    Path.home() / 'AppData' / 'Local' / 'piper' / 'voices' / 'en' / 'en_US' / 'amy' / 'medium' / f'{voice_name}.onnx',
]

found = False
for path in possible_paths:
    if path.exists():
        print(f'✅ 모델 발견: {path}')
        config_path = path.with_suffix('.onnx.json')
        if config_path.exists():
            print(f'✅ Config 발견: {config_path}')
        else:
            print(f'⚠️  Config 파일 없음: {config_path}')
        found = True
        break

if not found:
    print('❌ 모델을 찾을 수 없습니다.')
    print()
    print('다음 경로들을 확인했습니다:')
    for path in possible_paths:
        print(f'  - {path}')
    print()
    print('모델 다운로드:')
    print('  python download_all_models.py --auto')
else:
    print()
    print('=' * 60)
    print('✅ 모델이 정상적으로 설치되어 있습니다!')
    print('=' * 60)
"

echo.
pause
