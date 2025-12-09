"""Piper 영어 TTS 모델 빠른 다운로드 스크립트"""

import subprocess
from pathlib import Path

def download_english_tts():
    """영어 TTS 모델 다운로드"""
    print("=" * 60)
    print("Piper 영어 TTS 모델 다운로드")
    print("=" * 60)
    print()
    
    voice_name = "en_US-amy-medium"
    voice_path = "en/en_US/amy/medium"
    repo_id = "rhasspy/piper-voices"
    
    # 실제 파일명 (HuggingFace에서 확인)
    model_filename = "en_US-amy-medium.onnx"
    config_filename = "en_US-amy-medium.onnx.json"
    
    # 저장 디렉토리
    save_dir = Path.home() / ".local" / "share" / "piper" / "voices" / voice_path
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"음성: {voice_name}")
    print(f"저장 위치: {save_dir}")
    print()
    
    # 방법 1: huggingface_hub 사용
    try:
        from huggingface_hub import hf_hub_download
        
        print("HuggingFace Hub로 다운로드 중...")
        
        # 모델 파일 (정확한 파일명 사용)
        model_file = f"{voice_path}/{model_filename}"
        hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False
        )
        print(f"✅ {model_filename} 다운로드 완료")
        
        # Config 파일 (정확한 파일명 사용)
        config_file = f"{voice_path}/{config_filename}"
        hf_hub_download(
            repo_id=repo_id,
            filename=config_file,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False
        )
        print(f"✅ {config_filename} 다운로드 완료")
        
        print()
        print("=" * 60)
        print("✅ 다운로드 완료!")
        print("=" * 60)
        print(f"저장 위치: {save_dir}")
        return True
        
    except ImportError:
        print("⚠️  huggingface_hub가 없습니다. huggingface-cli로 시도합니다...")
        print()
    
    # 방법 2: huggingface-cli 사용
    try:
        print("huggingface-cli로 다운로드 중...")
        # 특정 경로의 파일만 다운로드 (정확한 파일명 사용)
        cmd = [
            "huggingface-cli", "download", repo_id,
            "--include", f"{voice_path}/{model_filename}",
            "--include", f"{voice_path}/{config_filename}",
            "--local-dir", str(save_dir.parent.parent.parent.parent),
            "--local-dir-use-symlinks", "False"
        ]
        result = subprocess.run(cmd, check=True, text=True)
        print()
        print("=" * 60)
        print("✅ 다운로드 완료!")
        print("=" * 60)
        print(f"저장 위치: {save_dir}")
        return True
    except FileNotFoundError:
        print("❌ huggingface-cli를 찾을 수 없습니다.")
        print()
        print("설치 방법:")
        print("  pip install huggingface-hub")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ 다운로드 실패: {e}")
        print()
        print("수동 다운로드 방법:")
        print(f"  https://huggingface.co/{repo_id}/tree/main/{voice_path}")
        return False

if __name__ == "__main__":
    try:
        download_english_tts()
    except KeyboardInterrupt:
        print("\n\n작업이 취소되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
