"""모든 필요한 모델을 찾고 순차적으로 다운로드하는 통합 스크립트"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import yaml

def check_model_exists(model_path: str) -> bool:
    """모델이 완전히 다운로드되었는지 확인"""
    if not model_path:
        return False
    
    path = Path(model_path)
    if path.exists() and path.is_dir():
        # 로컬 경로인 경우: 모델 파일과 config 파일 확인
        model_files = list(path.rglob("*.safetensors")) + \
                     list(path.rglob("*.bin")) + \
                     list(path.rglob("*.onnx")) + \
                     list(path.rglob("*.pt")) + \
                     list(path.rglob("*.pth"))
        
        # config.json도 확인 (모델이 완전한지 확인)
        config_file = path / "config.json"
        if len(model_files) > 0 and config_file.exists():
            return True
        return False
    
    # HuggingFace 캐시 확인
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        # 모델 ID에서 경로 추정
        if "/" in model_path:
            model_name = model_path.replace("/", "--")
            cache_path = cache_dir / f"models--{model_name}"
            if cache_path.exists():
                # 캐시 내에서 실제 모델 파일 찾기
                # HuggingFace 캐시 구조: models--{name}/snapshots/{hash}/
                snapshots_dir = cache_path / "snapshots"
                if snapshots_dir.exists():
                    # 가장 최근 스냅샷 확인
                    snapshots = list(snapshots_dir.iterdir())
                    if snapshots:
                        latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                        # 모델 파일 확인
                        model_files = list(latest_snapshot.rglob("*.safetensors")) + \
                                     list(latest_snapshot.rglob("*.bin")) + \
                                     list(latest_snapshot.rglob("*.pt")) + \
                                     list(latest_snapshot.rglob("*.pth"))
                        config_file = latest_snapshot / "config.json"
                        # 모델 파일과 config가 모두 있어야 완전히 다운로드된 것으로 간주
                        if len(model_files) > 0 and config_file.exists():
                            return True
                # snapshots가 없으면 아직 다운로드 중이거나 불완전
                return False
    
    return False

def check_whisper_model(model_name: str) -> bool:
    """Whisper 모델 확인"""
    try:
        import whisper
        cache_dir = whisper._MODELS
        model_path = os.path.join(cache_dir, f"{model_name}.pt")
        return os.path.exists(model_path)
    except:
        return False

def check_piper_voice(voice_name: str) -> bool:
    """Piper 음성 모델 확인 (기존 rhasspy 모델)"""
    # 영어 TTS 모델 (en_US-amy-medium) 특별 처리
    if voice_name == "en_US-amy-medium":
        # HuggingFace 캐시에서 먼저 확인
        hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        hf_model_dir = hf_cache_dir / "models--rhasspy--piper-voices"
        hf_paths = []
        if hf_model_dir.exists():
            snapshots_dir = hf_model_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                    hf_paths.append(latest_snapshot / "en" / "en_US" / "amy" / "medium" / f"{voice_name}.onnx")
                    hf_paths.append(latest_snapshot / "en" / "en_US" / "amy" / "medium" / "model.onnx")
        
        possible_paths = hf_paths + [
            Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium" / f"{voice_name}.onnx",
            Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium" / "model.onnx",
            Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / f"{voice_name}.onnx",
            Path("models") / "tts" / f"{voice_name}.onnx",
        ]
    else:
        possible_paths = [
            Path.home() / ".local" / "share" / "piper" / "voices" / voice_name.replace("-", "/") / "model.onnx",
            Path.home() / ".local" / "share" / "piper" / "voices" / voice_name / "model.onnx",
            Path("models") / "tts" / f"{voice_name}.onnx",
        ]
    
    for path in possible_paths:
        if path.exists():
            return True
    return False

def check_piper_voice_hf(model_id: str) -> bool:
    """HuggingFace Piper 음성 모델 확인"""
    # 파일명 추출
    if model_id == "neurlang/piper-onnx-kss-korean":
        model_file = "piper-kss-korean.onnx"
        config_file = "piper-kss-korean.onnx.json"
    else:
        # 일반적인 경우
        model_name = model_id.split("/")[-1]
        file_prefix = model_name.replace("piper-onnx-", "piper-")
        model_file = f"{file_prefix}.onnx"
        config_file = f"{file_prefix}.onnx.json"
    
    # 저장 디렉토리 확인
    save_dir = Path("models") / "tts" / model_id.replace("/", "_")
    model_path = save_dir / model_file
    config_path = save_dir / config_file
    
    if model_path.exists() and config_path.exists():
        return True
    
    # HuggingFace 캐시 확인
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists():
        model_name_safe = model_id.replace("/", "--")
        cache_model_dir = cache_dir / f"models--{model_name_safe}"
        if cache_model_dir.exists():
            # 캐시에서 파일 찾기
            cached_files = list(cache_model_dir.rglob("*.onnx"))
            if cached_files:
                return True
    
    return False

def load_config() -> dict:
    """config.yaml 로드"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

def scan_required_models() -> Dict[str, Dict]:
    """필요한 모든 모델 스캔"""
    config = load_config()
    
    models = {}
    
    # 1. LLM 모델
    llm_config = config.get("llm", {})
    llm_model = llm_config.get("model", "llama-3.1-8b")
    llm_path = llm_config.get("model_path")
    
    if llm_model == "llama-3.1-8b":
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        local_path = llm_path or "models/llm/llama-3.1-8b"
    elif llm_model == "deepseek-r1-7b":
        model_id = "deepseek-ai/DeepSeek-R1"
        local_path = llm_path or "models/llm/deepseek-r1-7b"
    else:
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        local_path = llm_path or "models/llm/llama-3.1-8b"
    
    models["llm"] = {
        "name": f"LLM ({llm_model})",
        "model_id": model_id,
        "local_path": local_path,
        "size": "~16GB",
        "exists": check_model_exists(local_path) or check_model_exists(model_id),
        "required": True
    }
    
    # 2. Whisper 모델
    whisper_config = config.get("whisper", {})
    whisper_model = whisper_config.get("model", "openai/whisper-large-v3")
    
    # HuggingFace 모델 ID인지 확인
    is_hf_model = "/" in whisper_model
    
    if is_hf_model:
        # HuggingFace 모델
        models["whisper"] = {
            "name": f"Whisper ({whisper_model})",
            "model_id": whisper_model,
            "local_path": None,  # HuggingFace 캐시
            "size": "~3GB",
            "exists": check_model_exists(whisper_model),
            "required": True
        }
    else:
        # 기존 whisper 라이브러리 모델
        models["whisper"] = {
            "name": f"Whisper ({whisper_model})",
            "model_id": whisper_model,
            "local_path": None,  # Whisper는 자동 캐시
            "size": "~3GB",
            "exists": check_whisper_model(whisper_model),
            "required": True
        }
    
    # 3. TTS 모델
    tts_config = config.get("tts", {})
    tts_model = tts_config.get("model", "piper")
    
    if tts_model == "piper":
        # 한국어 음성
        ko_voice = tts_config.get("piper", {}).get("voice", "neurlang/piper-onnx-kss-korean")
        
        # HuggingFace 모델 ID인지 확인
        is_hf_model = "/" in ko_voice and ko_voice.count("/") == 1
        
        if is_hf_model:
            # HuggingFace 모델
            models["tts_korean"] = {
                "name": f"Piper 한국어 ({ko_voice})",
                "model_id": ko_voice,  # "neurlang/piper-onnx-kss-korean"
                "local_path": None,
                "size": "~10MB",
                "exists": check_piper_voice_hf(ko_voice),
                "required": True,
                "is_huggingface": True
            }
        else:
            # 기존 rhasspy 모델
            models["tts_korean"] = {
                "name": f"Piper 한국어 ({ko_voice})",
                "model_id": f"rhasspy/piper-voices/{ko_voice}",
                "local_path": None,
                "size": "~10MB",
                "exists": check_piper_voice(ko_voice),
                "required": True,
                "is_huggingface": False
            }
        
        # 영어 음성 (비디오 생성용)
        # HuggingFace에서 직접 다운로드: rhasspy/piper-voices의 en/en_US/amy/medium 경로
        models["tts_english"] = {
            "name": "Piper 영어 (en_US-amy-medium)",
            "model_id": "rhasspy/piper-voices",  # 전체 리포지토리
            "local_path": None,
            "size": "~10MB",
            "exists": check_piper_voice("en_US-amy-medium"),
            "required": True,
            "voice_name": "en_US-amy-medium",  # 실제 음성 이름
            "voice_path": "en/en_US/amy/medium"  # HuggingFace 내 경로
        }
    elif tts_model == "vibevoice":
        vibevoice_id = tts_config.get("vibevoice", {}).get("model_id", "microsoft/VibeVoice-1.5B")
        models["tts_vibevoice"] = {
            "name": "VibeVoice-1.5B",
            "model_id": vibevoice_id,
            "local_path": None,
            "size": "~5.4GB",
            "exists": check_model_exists(vibevoice_id),
            "required": True
        }
    
    # 4. Stable Diffusion XL
    video_gen_config = config.get("video_generation", {})
    if video_gen_config.get("model") == "svd" or video_gen_config.get("use_image_generation"):
        models["sdxl"] = {
            "name": "Stable Diffusion XL",
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "local_path": None,
            "size": "~7GB",
            "exists": check_model_exists("stabilityai/stable-diffusion-xl-base-1.0"),
            "required": True
        }
        
        # 5. Stable Video Diffusion
        models["svd"] = {
            "name": "Stable Video Diffusion",
            "model_id": "stabilityai/stable-video-diffusion-img2vid",
            "local_path": None,
            "size": "~17GB",
            "exists": check_model_exists("stabilityai/stable-video-diffusion-img2vid"),
            "required": True
        }
    
    return models

def download_model(model_info: Dict, index: int, total: int) -> bool:
    """단일 모델 다운로드"""
    name = model_info["name"]
    model_id = model_info["model_id"]
    local_path = model_info.get("local_path")
    size = model_info.get("size", "알 수 없음")
    
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] {name} 다운로드 중...")
    print(f"크기: {size}")
    print(f"{'='*60}\n")
    
    try:
        # HuggingFace CLI 명령어 구성
        cmd = ["huggingface-cli", "download", model_id]
        
        if local_path:
            Path(local_path).mkdir(parents=True, exist_ok=True)
            cmd.extend(["--local-dir", local_path, "--local-dir-use-symlinks", "False"])


        
        # 다운로드 실행
        result = subprocess.run(
            cmd,
            check=True,
            text=True
        )
        
        # 저장 위치 표시
        if local_path:
            save_location = str(Path(local_path).absolute())
        else:
            # HuggingFace 기본 캐시 위치
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_name_safe = model_id.replace("/", "--")
            save_location = str(cache_dir / f"models--{model_name_safe}")
        
        print(f"\n✅ {name} 다운로드 완료!")
        print(f"📁 저장 위치: {save_location}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {name} 다운로드 실패: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n\n⚠️  {name} 다운로드가 중단되었습니다.")
        print("나중에 다시 실행하면 이어서 다운로드할 수 있습니다.")
        return False
    except FileNotFoundError:
        print("\n❌ huggingface-cli를 찾을 수 없습니다.")
        print("다음 명령어로 설치하세요:")
        print("  pip install huggingface-hub")
        return False

def download_whisper_model(model_name: str, index: int, total: int) -> bool:
    """Whisper 모델 다운로드 (HuggingFace CLI 사용)"""
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] Whisper ({model_name}) 다운로드 중...")
    print(f"크기: ~3GB")
    print(f"{'='*60}\n")
    
    # HuggingFace 모델 ID인 경우
    if "/" in model_name:
        try:
            cmd = ["huggingface-cli", "download", model_name]
            result = subprocess.run(cmd, check=True, text=True)
            
            # 저장 위치 표시 (HuggingFace 기본 캐시)
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_name_safe = model_name.replace("/", "--")
            save_location = str(cache_dir / f"models--{model_name_safe}")
            
            print(f"\n✅ Whisper ({model_name}) 다운로드 완료!")
            print(f"📁 저장 위치: {save_location}")
            return True
        except FileNotFoundError:
            print("\n❌ huggingface-cli를 찾을 수 없습니다.")
            print("다음 명령어로 설치하세요:")
            print("  pip install huggingface-hub")
            return False
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Whisper 다운로드 실패: {e}")
            return False
    else:
        # 기존 whisper 라이브러리 모델 (fallback)
        try:
            import whisper
            print("Whisper 모델 로드 중 (자동 다운로드)...")
            model = whisper.load_model(model_name)
            print(f"\n✅ Whisper ({model_name}) 다운로드 완료!")
            return True
        except Exception as e:
            print(f"\n❌ Whisper 다운로드 실패: {e}")
            return False

def download_piper_voice_english(voice_name: str, voice_path: str, index: int, total: int) -> bool:
    """Piper 영어 음성 모델 다운로드 (HuggingFace CLI 사용)"""
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] Piper 영어 음성 ({voice_name}) 다운로드 중...")
    print(f"크기: ~10MB")
    print(f"{'='*60}\n")
    
    try:
        from huggingface_hub import hf_hub_download
        
        repo_id = "rhasspy/piper-voices"
        # 정확한 파일명 사용 (en_US-amy-medium.onnx, en_US-amy-medium.onnx.json)
        model_filename = f"{voice_name}.onnx"
        config_filename = f"{voice_name}.onnx.json"
        model_file = f"{voice_path}/{model_filename}"
        config_file = f"{voice_path}/{config_filename}"
        
        # 저장 디렉토리 (Piper 표준 경로)
        save_dir = Path.home() / ".local" / "share" / "piper" / "voices" / voice_path
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"HuggingFace에서 다운로드 중: {repo_id}/{model_file}")
        
        # 모델 파일 다운로드
        print(f"모델 파일 다운로드 중: {model_filename}")
        print(f"저장 디렉토리: {save_dir}")
        
        downloaded_model = hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False
        )
        
        # 다운로드된 파일 경로 확인
        downloaded_model_path = Path(downloaded_model)
        target_model_path = save_dir / model_filename
        
        print(f"다운로드된 파일 경로: {downloaded_model_path}")
        print(f"목표 파일 경로: {target_model_path}")
        
        # 파일이 다른 위치에 다운로드된 경우 이동
        if downloaded_model_path != target_model_path:
            if downloaded_model_path.exists() and not target_model_path.exists():
                import shutil
                shutil.move(str(downloaded_model_path), str(target_model_path))
                print(f"✅ 모델 파일 이동 완료: {target_model_path}")
            elif target_model_path.exists():
                print(f"✅ 모델 파일 이미 존재: {target_model_path}")
            else:
                print(f"⚠️  파일 위치 확인 필요: {downloaded_model_path}")
        else:
            print(f"✅ 모델 파일 다운로드 완료: {target_model_path}")
        
        # 최종 파일 존재 확인
        if target_model_path.exists():
            print(f"✅ 최종 확인: 모델 파일 존재 - {target_model_path}")
        else:
            print(f"⚠️  경고: 목표 경로에 파일 없음 - {target_model_path}")
            if downloaded_model_path.exists():
                print(f"   대신 이 경로에 있음: {downloaded_model_path}")
        
        # Config 파일 다운로드
        print(f"Config 파일 다운로드 중: {config_filename}")
        downloaded_config = hf_hub_download(
            repo_id=repo_id,
            filename=config_file,
            local_dir=str(save_dir),
            local_dir_use_symlinks=False
        )
        
        # 다운로드된 파일을 올바른 위치로 이동 (필요시)
        downloaded_config_path = Path(downloaded_config)
        target_config_path = save_dir / config_filename
        if downloaded_config_path != target_config_path and downloaded_config_path.exists():
            # 파일이 다른 위치에 다운로드된 경우 이동
            if not target_config_path.exists():
                import shutil
                shutil.move(str(downloaded_config_path), str(target_config_path))
                print(f"✅ Config 파일 이동 완료: {target_config_path}")
            else:
                print(f"✅ Config 파일 다운로드 완료: {target_config_path}")
        else:
            print(f"✅ Config 파일 다운로드 완료: {downloaded_config_path}")
        
        # 최종 확인
        final_model_path = target_model_path if target_model_path.exists() else downloaded_model_path
        final_config_path = target_config_path if target_config_path.exists() else downloaded_config_path
        
        print(f"\n✅ Piper 영어 음성 ({voice_name}) 다운로드 완료!")
        print(f"저장 위치: {save_dir}")
        print(f"모델 파일: {final_model_path}")
        print(f"  존재 여부: {final_model_path.exists()}")
        print(f"Config 파일: {final_config_path}")
        print(f"  존재 여부: {final_config_path.exists()}")
        
        if not final_model_path.exists():
            print(f"\n⚠️  경고: 모델 파일을 찾을 수 없습니다!")
            print(f"다운로드된 파일 확인: {downloaded_model_path}")
        
        return True
        
    except ImportError:
        print("\n⚠️  huggingface_hub 패키지가 없습니다.")
        print("huggingface-cli로 대체 다운로드 시도 중...")
        try:
            # huggingface-cli로 특정 파일 다운로드
            repo_id = "rhasspy/piper-voices"
            save_dir = Path.home() / ".local" / "share" / "piper" / "voices" / voice_path
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 전체 리포지토리 다운로드 (일부만 필요하지만)
            cmd = ["huggingface-cli", "download", repo_id, "--include", f"{voice_path}/*", "--local-dir", str(save_dir.parent.parent.parent.parent), "--local-dir-use-symlinks", "False"]
            result = subprocess.run(cmd, check=True, text=True)
            
            print(f"\n✅ Piper 영어 음성 ({voice_name}) 다운로드 완료!")
            print(f"저장 위치: {save_dir}")
            return True
        except Exception as e:
            print(f"\n❌ 다운로드 실패: {e}")
            print(f"\n수동 다운로드 방법:")
            print(f"  https://huggingface.co/{repo_id}/tree/main/{voice_path}")
            return False
    except Exception as e:
        print(f"\n❌ HuggingFace 다운로드 실패: {e}")
        return False

def download_piper_voice(voice_name: str, index: int, total: int, is_huggingface: bool = False) -> bool:
    """Piper 음성 모델 다운로드 (HuggingFace CLI 사용)"""
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] Piper 음성 ({voice_name}) 다운로드 중...")
    print(f"크기: ~10MB")
    print(f"{'='*60}\n")
    
    # HuggingFace 모델인 경우
    if is_huggingface or "/" in voice_name:
        try:
            model_id = voice_name
            print(f"HuggingFace에서 다운로드 중: {model_id}")
            
            # 저장 디렉토리
            save_dir = Path("models") / "tts" / model_id.replace("/", "_")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # huggingface-cli로 다운로드
            cmd = ["huggingface-cli", "download", model_id, "--local-dir", str(save_dir), "--local-dir-use-symlinks", "False"]
            result = subprocess.run(cmd, check=True, text=True)
            
            print(f"\n✅ Piper 음성 ({voice_name}) 다운로드 완료!")
            print(f"저장 위치: {save_dir}")
            return True
            
        except FileNotFoundError:
            print("\n❌ huggingface-cli를 찾을 수 없습니다.")
            print("다음 명령어로 설치하세요:")
            print("  pip install huggingface-hub")
            return False
        except Exception as e:
            print(f"\n❌ HuggingFace 다운로드 실패: {e}")
            return False
    else:
        # 기존 rhasspy 모델
        try:
            # piper 명령어로 다운로드 시도
            cmd = ["piper", "download", "--voice", voice_name]
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            print(f"\n✅ Piper 음성 ({voice_name}) 다운로드 완료!")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            # 수동 다운로드 안내
            print(f"\n⚠️  Piper 명령어를 사용할 수 없습니다.")
            print(f"수동 다운로드:")
            print(f"  https://huggingface.co/rhasspy/piper-voices/tree/main/{voice_name}")
            print(f"또는:")
            print(f"  python -m piper.download --voice {voice_name}")
            return False

def main():
    """메인 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="모델 스캔 및 다운로드 도구")
    parser.add_argument(
        "--auto", "-y",
        action="store_true",
        help="자동 모드: 모든 확인 없이 자동으로 다운로드 진행"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="강제 모드: 이미 다운로드된 모델도 다시 다운로드"
    )
    args = parser.parse_args()
    auto_mode = args.auto
    force_mode = args.force
    
    print("=" * 60)
    print("모델 스캔 및 다운로드 도구")
    if auto_mode:
        print("🤖 자동 모드: 모든 확인 없이 자동 진행")
    if force_mode:
        print("🔄 강제 모드: 이미 다운로드된 모델도 다시 다운로드")
    print("=" * 60)
    
    print("\n필요한 모델을 스캔 중...\n")
    
    # 모델 스캔
    models = scan_required_models()
    
    # 없는 모델 필터링 (강제 모드면 모든 모델을 다운로드 대상으로)
    if force_mode:
        missing_models = models
        existing_models = {}
    else:
        missing_models = {k: v for k, v in models.items() if not v["exists"]}
        existing_models = {k: v for k, v in models.items() if v["exists"]}
    
    # 결과 출력
    print("=" * 60)
    print("모델 상태")
    print("=" * 60)
    
    if existing_models:
        print("\n✅ 이미 설치된 모델:")
        for key, info in existing_models.items():
            print(f"  ✓ {info['name']}")
    
    if missing_models:
        print(f"\n❌ 다운로드 필요한 모델 ({len(missing_models)}개):")
        for i, (key, info) in enumerate(missing_models.items(), 1):
            print(f"  {i}. {info['name']} ({info['size']})")
    else:
        print("\n✅ 모든 필요한 모델이 이미 설치되어 있습니다!")
        return
    
    # 다운로드 확인 및 선택
    if missing_models:
        print("\n" + "=" * 60)
        total_size = sum(
            float(info["size"].replace("~", "").replace("GB", "").replace("MB", "")) 
            for info in missing_models.values() 
            if "GB" in info["size"] or "MB" in info["size"]
        )
        print(f"예상 총 다운로드 크기: ~{total_size:.1f}GB")
        print("=" * 60)
        
        if auto_mode:
            # 자동 모드: 모든 모델 자동 다운로드
            print("\n🤖 자동 모드: 모든 모델을 자동으로 다운로드합니다.")
            selected_models = missing_models
        else:
            print("\n다운로드 옵션:")
            print("1. 모든 모델 다운로드 (자동)")
            print("2. 모델별 선택 다운로드")
            print("3. 취소")
            
            choice = input("\n선택 (1/2/3): ").strip()
            
            if choice == "3":
                print("\n다운로드가 취소되었습니다.")
                return
            elif choice == "2":
                # 모델별 선택
                selected_models = {}
                print("\n다운로드할 모델을 선택하세요 (번호 입력, 여러 개는 쉼표로 구분, 'all'은 모두 선택):")
                for i, (key, info) in enumerate(missing_models.items(), 1):
                    print(f"  {i}. {info['name']} ({info['size']})")
                
                selection = input("\n선택: ").strip()
                
                if selection.lower() == "all":
                    selected_models = missing_models
                else:
                    try:
                        indices = [int(x.strip()) for x in selection.split(",")]
                        model_list = list(missing_models.items())
                        for idx in indices:
                            if 1 <= idx <= len(model_list):
                                key, info = model_list[idx - 1]
                                selected_models[key] = info
                    except ValueError:
                        print("❌ 잘못된 입력입니다.")
                        return
                    
                    if not selected_models:
                        print("❌ 선택된 모델이 없습니다.")
                        return
            else:
                # 모든 모델 다운로드
                selected_models = missing_models
        
        if not selected_models:
            print("\n다운로드할 모델이 없습니다.")
            return
        
        print(f"\n선택된 모델 ({len(selected_models)}개):")
        for info in selected_models.values():
            print(f"  - {info['name']} ({info['size']})")
        
        if not auto_mode:
            response = input("\n다운로드를 시작하시겠습니까? (y/n): ").strip().lower()
            if response != 'y':
                print("\n다운로드가 취소되었습니다.")
                return
        else:
            print("\n🤖 자동 모드: 다운로드를 시작합니다...")
        
        # selected_models를 missing_models로 업데이트
        missing_models = selected_models
        
        # 순차적으로 다운로드
        print("\n" + "=" * 60)
        print("다운로드 시작")
        print("=" * 60)
        print("\n⚠️  중단(Ctrl+C)해도 나중에 다시 실행하면 이어서 다운로드할 수 있습니다.\n")
        
        total = len(missing_models)
        success_count = 0
        failed_models = []
        
        for i, (key, info) in enumerate(missing_models.items(), 1):
            try:
                if key == "whisper":
                    success = download_whisper_model(info["model_id"], i, total)
                elif key.startswith("tts_") and "piper" in key:
                    # 영어 음성의 경우 특별 처리
                    if key == "tts_english":
                        voice_name = info.get("voice_name", "en_US-amy-medium")
                        voice_path = info.get("voice_path", "en/en_US/amy/medium")
                        success = download_piper_voice_english(voice_name, voice_path, i, total)
                    else:
                        voice_name = info["model_id"]
                        is_hf = info.get("is_huggingface", False) or "/" in voice_name
                        success = download_piper_voice(voice_name, i, total, is_huggingface=is_hf)
                elif key == "tts_vibevoice":
                    # VibeVoice는 일반 HuggingFace 모델
                    success = download_model(info, i, total)
                else:
                    success = download_model(info, i, total)
                
                if success:
                    success_count += 1
                else:
                    failed_models.append(info['name'])
                    print(f"\n⚠️  {info['name']} 다운로드 실패")
                    if auto_mode:
                        print("🤖 자동 모드: 계속 진행합니다...")
                    else:
                        response = input("계속 진행하시겠습니까? (y/n): ").strip().lower()
                        if response != 'y':
                            print("\n다운로드가 중단되었습니다.")
                            break
            except KeyboardInterrupt:
                print("\n\n⚠️  사용자에 의해 중단되었습니다.")
                break
            except Exception as e:
                print(f"\n❌ {info['name']} 다운로드 중 오류: {e}")
                failed_models.append(info['name'])
                if auto_mode:
                    print("🤖 자동 모드: 계속 진행합니다...")
                else:
                    response = input("계속 진행하시겠습니까? (y/n): ").strip().lower()
                    if response != 'y':
                        break
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("다운로드 완료")
        print("=" * 60)
        print(f"\n성공: {success_count}/{total}")
        print(f"실패: {total - success_count}/{total}")
        
        if failed_models:
            print("\n실패한 모델:")
            for name in failed_models:
                print(f"  - {name}")
        
        if success_count == total:
            print("\n✅ 모든 모델 다운로드 완료!")
            print("\n이제 프로그램을 실행할 수 있습니다:")
            print("  python main.py")
        else:
            print("\n⚠️  일부 모델 다운로드에 실패했습니다.")
            print("나중에 다시 실행하여 남은 모델을 다운로드할 수 있습니다:")
            print("  python download_all_models.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n작업이 취소되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
