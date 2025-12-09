"""다운로드된 모델 확인 스크립트"""

import os
from pathlib import Path
import sys

def check_model(model_path: str, model_name: str):
    """모델 디렉토리 확인"""
    print(f"\n{'='*60}")
    print(f"모델 확인: {model_name}")
    print(f"경로: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 디렉토리가 존재하지 않습니다: {model_path}")
        return False
    
    # 필수 파일 확인
    required_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    
    # safetensors 또는 pytorch 모델 파일 확인
    model_files = [
        "model.safetensors.index.json",  # safetensors 분할 모델
        "model.safetensors",  # 단일 safetensors 파일
        "pytorch_model.bin.index.json",  # pytorch 분할 모델
        "pytorch_model.bin",  # 단일 pytorch 모델
    ]
    
    print("\n[필수 설정 파일]")
    all_required = True
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - 누락됨")
            all_required = False
    
    print("\n[모델 파일]")
    model_found = False
    total_size = 0
    
    # safetensors.index.json 확인
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        print(f"✅ model.safetensors.index.json 발견")
        model_found = True
        
        # 분할 파일 확인
        import json
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            weight_map = index_data.get("weight_map", {})
            unique_files = set(weight_map.values())
            
            print(f"\n[분할 모델 파일] ({len(unique_files)}개 파일)")
            for i, filename in enumerate(sorted(unique_files), 1):
                file_path = os.path.join(model_path, filename)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    total_size += size
                    size_gb = size / (1024**3)
                    print(f"✅ {filename} ({size_gb:.2f} GB)")
                else:
                    print(f"❌ {filename} - 누락됨")
                    model_found = False
        except Exception as e:
            print(f"⚠️  index.json 파싱 오류: {e}")
    
    # 단일 모델 파일 확인
    for file in model_files[1:]:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            size_gb = size / (1024**3)
            print(f"✅ {file} ({size_gb:.2f} GB)")
            model_found = True
            break
    
    if not model_found:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        print("   다음 중 하나가 있어야 합니다:")
        for file in model_files:
            print(f"   - {file}")
        return False
    
    # 전체 크기 표시
    if total_size > 0:
        total_gb = total_size / (1024**3)
        print(f"\n📦 전체 모델 크기: {total_gb:.2f} GB")
    
    # 기타 파일 확인
    print("\n[기타 파일]")
    other_files = [
        "generation_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    
    for file in other_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file} ({size:,} bytes)")
    
    # 결과 요약
    print(f"\n{'='*60}")
    if all_required and model_found:
        print("✅ 모델이 정상적으로 다운로드되었습니다!")
        return True
    else:
        print("❌ 모델이 불완전합니다. 다운로드를 완료하세요.")
        return False

def main():
    print("="*60)
    print("모델 다운로드 확인")
    print("="*60)
    
    # config.yaml에서 모델 경로 확인
    config_path = "config.yaml"
    model_paths = []
    
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            llm_config = config.get("llm", {})
            model_path = llm_config.get("model_path")
            model_name = llm_config.get("model", "deepseek-r1-7b")
            
            if model_path:
                model_paths.append((model_path, model_name))
            else:
                # 기본 경로 확인
                if model_name == "deepseek-r1-7b":
                    model_paths.append(("models/llm/deepseek-r1-7b", "DeepSeek-R1 7B"))
                elif model_name == "llama-3.1-8b":
                    model_paths.append(("models/llm/llama-3.1-8b", "Llama 3.1 8B"))
        except Exception as e:
            print(f"⚠️  config.yaml 읽기 오류: {e}")
    
    # 기본 경로들 확인
    default_paths = [
        ("models/llm/deepseek-r1-7b", "DeepSeek-R1 7B"),
        ("models/llm/llama-3.1-8b", "Llama 3.1 8B"),
    ]
    
    # 중복 제거
    all_paths = {}
    for path, name in model_paths + default_paths:
        if path not in all_paths:
            all_paths[path] = name
    
    if not all_paths:
        print("\n❌ 확인할 모델 경로를 찾을 수 없습니다.")
        print("\n수동으로 확인하려면:")
        print("python check_model.py <모델_경로>")
        return 1
    
    # 각 모델 확인
    all_ok = True
    for model_path, model_name in all_paths.items():
        if os.path.exists(model_path):
            if not check_model(model_path, model_name):
                all_ok = False
        else:
            print(f"\n⚠️  모델 경로가 존재하지 않습니다: {model_path}")
    
    # 요약
    print("\n" + "="*60)
    if all_ok and any(os.path.exists(path) for path in all_paths.keys()):
        print("✅ 모든 모델이 정상입니다!")
        return 0
    else:
        print("❌ 일부 모델이 누락되었거나 불완전합니다.")
        return 1

if __name__ == "__main__":
    # 명령줄 인자로 경로 지정 가능
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) > 2 else "Custom Model"
        success = check_model(model_path, model_name)
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())

