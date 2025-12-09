"""
영어 TTS 모델 위치 찾기 스크립트
"""
from pathlib import Path
import sys

def find_tts_model():
    """영어 TTS 모델 찾기"""
    voice_name = "en_US-amy-medium"
    
    print("=" * 60)
    print("영어 TTS 모델 검색 중...")
    print("=" * 60)
    print()
    
    # HuggingFace 캐시 확인
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    hf_model_dir = hf_cache_dir / "models--rhasspy--piper-voices"
    
    print(f"HuggingFace 캐시 디렉토리: {hf_cache_dir}")
    print(f"  존재 여부: {hf_cache_dir.exists()}")
    print(f"Piper 모델 디렉토리: {hf_model_dir}")
    print(f"  존재 여부: {hf_model_dir.exists()}")
    print()
    
    found_paths = []
    
    if hf_model_dir.exists():
        print("✅ HuggingFace 캐시 발견!")
        snapshots_dir = hf_model_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            print(f"  스냅샷 개수: {len(snapshots)}")
            
            for snapshot in snapshots:
                print(f"  스냅샷: {snapshot.name}")
                
                # 여러 경로 확인
                possible_paths = [
                    snapshot / "en" / "en_US" / "amy" / "medium" / f"{voice_name}.onnx",
                    snapshot / "en" / "en_US" / "amy" / "medium" / "model.onnx",
                    snapshot / "en" / "en_US" / f"{voice_name}.onnx",
                    snapshot / "en" / f"{voice_name}.onnx",
                    snapshot / f"{voice_name}.onnx",
                ]
                
                for path in possible_paths:
                    if path.exists():
                        print(f"    ✅ 발견: {path}")
                        found_paths.append(path)
                        
                        # Config 파일도 확인
                        config_path = path.with_suffix(".onnx.json")
                        if config_path.exists():
                            print(f"    ✅ Config: {config_path}")
                        else:
                            # 다른 위치에서 config 찾기
                            config_candidates = [
                                path.parent / f"{voice_name}.onnx.json",
                                path.parent / "model.onnx.json",
                            ]
                            for candidate in config_candidates:
                                if candidate.exists():
                                    print(f"    ✅ Config (다른 위치): {candidate}")
                                    break
    else:
        print("❌ HuggingFace 캐시에서 모델을 찾을 수 없습니다.")
        print()
        print("다음 경로들을 확인했습니다:")
        print(f"  - {hf_model_dir}")
    
    # 다른 가능한 경로들도 확인
    print()
    print("=" * 60)
    print("기타 경로 확인")
    print("=" * 60)
    print()
    
    other_paths = [
        Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium" / f"{voice_name}.onnx",
        Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium" / "model.onnx",
        Path.home() / ".local" / "share" / "piper" / "voices" / "en" / "en_US" / f"{voice_name}.onnx",
        Path("models") / "tts" / f"{voice_name}.onnx",
        Path.home() / "AppData" / "Local" / "piper" / "voices" / "en" / "en_US" / "amy" / "medium" / f"{voice_name}.onnx",
    ]
    
    for path in other_paths:
        if path.exists():
            print(f"✅ 발견: {path}")
            found_paths.append(path)
        else:
            print(f"❌ 없음: {path}")
    
    print()
    print("=" * 60)
    if found_paths:
        print(f"✅ 총 {len(found_paths)}개의 모델 파일을 찾았습니다!")
        print()
        print("사용 가능한 모델:")
        for i, path in enumerate(found_paths, 1):
            print(f"  {i}. {path}")
    else:
        print("❌ 모델을 찾을 수 없습니다.")
        print()
        print("모델 다운로드:")
        print("  python download_all_models.py --auto")
    print("=" * 60)

if __name__ == "__main__":
    find_tts_model()
