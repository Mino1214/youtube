"""LLM 모델 수동 다운로드 스크립트 (더 안정적)"""

import os
import sys
from pathlib import Path

def download_model_manual():
    """HuggingFace CLI를 사용하여 모델 수동 다운로드"""
    
    print("=" * 60)
    print("LLM 모델 수동 다운로드")
    print("=" * 60)
    print("\n이 방법은 더 안정적이고 재개 가능합니다.")
    print("중단되어도 다시 실행하면 이어서 다운로드됩니다.\n")
    
    # 모델 선택
    print("다운로드할 모델을 선택하세요:")
    print("1. DeepSeek-R1 7B (deepseek-ai/DeepSeek-R1)")
    print("2. Llama 3.1 8B (meta-llama/Llama-3.1-8B-Instruct)")
    
    choice = input("\n선택 (1 또는 2): ").strip()
    
    if choice == "1":
        model_id = "deepseek-ai/DeepSeek-R1"
        local_dir = "models/llm/deepseek-r1-7b"
    elif choice == "2":
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        local_dir = "models/llm/llama-3.1-8b"
    else:
        print("❌ 잘못된 선택입니다.")
        return False
    
    print(f"\n모델: {model_id}")
    print(f"저장 위치: {local_dir}")
    
    # 디렉토리 생성
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # HuggingFace CLI 명령어
    cmd = f'huggingface-cli download {model_id} --local-dir {local_dir} --local-dir-use-symlinks False'
    
    print("\n다음 명령어를 실행하세요:")
    print("=" * 60)
    print(cmd)
    print("=" * 60)
    print("\n또는 자동으로 실행하시겠습니까? (y/n): ", end="")
    
    auto = input().strip().lower()
    
    if auto == 'y':
        try:
            import subprocess
            print("\n다운로드 시작...")
            print("⚠️  이 작업은 시간이 오래 걸릴 수 있습니다 (수십 GB).")
            print("⚠️  중단(Ctrl+C)해도 나중에 다시 실행하면 이어서 다운로드됩니다.\n")
            
            result = subprocess.run(
                cmd.split(),
                check=True
            )
            
            print("\n✅ 다운로드 완료!")
            print(f"\nconfig.yaml에서 model_path를 설정하세요:")
            print(f"llm:")
            print(f"  model_path: \"{local_dir}\"")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 다운로드 실패: {e}")
            print("\n수동으로 다음 명령어를 실행하세요:")
            print(cmd)
            return False
        except KeyboardInterrupt:
            print("\n\n⚠️  다운로드가 중단되었습니다.")
            print("나중에 같은 명령어를 다시 실행하면 이어서 다운로드됩니다.")
            return False
    else:
        print("\n위 명령어를 직접 실행하세요.")
        return False

if __name__ == "__main__":
    try:
        download_model_manual()
    except KeyboardInterrupt:
        print("\n\n작업이 취소되었습니다.")
        sys.exit(1)

