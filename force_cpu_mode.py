"""CPU 모드 강제 설정 스크립트"""

import yaml

# config.yaml 읽기
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# CPU 모드로 강제 설정
config['whisper']['device'] = 'cpu'
config['llm']['use_gpu'] = False

# 저장
with open('config.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

print("✅ config.yaml이 CPU 모드로 강제 설정되었습니다.")
print("\n변경 사항:")
print("  - whisper.device: cpu")
print("  - llm.use_gpu: false")
print("\n이제 python main.py를 실행하세요!")

