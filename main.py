#!/usr/bin/env python3
"""간단한 진입점 - python main.py로 비디오 생성"""

import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import VideoConversionPipeline

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def list_available_channels():
    """사용 가능한 채널 목록 반환"""
    try:
        channels_path = Path("channels.yaml")
        if not channels_path.exists():
            return {}
        
        import yaml
        with open(channels_path, 'r', encoding='utf-8') as f:
            channels_config = yaml.safe_load(f)
        
        return channels_config.get("channels", {})
    except Exception as e:
        logger.warning(f"채널 목록 로드 실패: {e}")
        return {}


def select_channel():
    """채널 선택 인터페이스"""
    channels = list_available_channels()
    
    if not channels:
        logger.info("채널 프로필이 없습니다. 기본 설정을 사용합니다.")
        return None
    
    print("\n" + "=" * 60)
    print("채널 선택")
    print("=" * 60)
    print("\n사용 가능한 채널:")
    
    channel_list = list(channels.items())
    for i, (channel_id, channel_info) in enumerate(channel_list, 1):
        name = channel_info.get("name", channel_id)
        desc = channel_info.get("description", "")
        print(f"  {i}. {name} ({channel_id})")
        if desc:
            print(f"     └─ {desc}")
    
    print(f"  {len(channel_list) + 1}. 기본 설정 사용")
    
    while True:
        try:
            choice = input(f"\n선택 (1-{len(channel_list) + 1}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(channel_list):
                selected_id = channel_list[choice_num - 1][0]
                selected_name = channel_list[choice_num - 1][1].get("name", selected_id)
                logger.info(f"선택된 채널: {selected_name} ({selected_id})")
                return selected_id
            elif choice_num == len(channel_list) + 1:
                logger.info("기본 설정을 사용합니다.")
                return None
            else:
                print(f"❌ 1-{len(channel_list) + 1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("❌ 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n\n작업이 취소되었습니다.")
            return None


def main():
    """메인 함수 - 텍스트에서 비디오 생성"""
    
    # 채널 선택
    selected_channel = select_channel()
    
    # 예제 영어 텍스트 (사용자가 수정 가능)
    english_text = """
    Welcome to our video. Today we will discuss an important topic.
    Artificial intelligence is transforming the world around us.
    From healthcare to education, AI is making a significant impact.
    Let's explore how this technology is changing our daily lives.
    """
    
    # 또는 파일에서 읽기 (input.txt 또는 input_text.txt)
    text_file = None
    for filename in ["input.txt", "input_text.txt"]:
        candidate = Path(filename)
        if candidate.exists():
            text_file = candidate
            break
    
    if text_file:
        logger.info(f"텍스트 파일에서 읽기: {text_file}")
        with open(text_file, 'r', encoding='utf-8') as f:
            english_text = f.read().strip()
    
    if not english_text or not english_text.strip():
        logger.error("입력 텍스트가 없습니다!")
        logger.info("다음 중 하나를 선택하세요:")
        logger.info("1. main.py 파일의 english_text 변수를 수정")
        logger.info("2. input_text.txt 파일을 생성하고 텍스트 입력")
        return 1
    
    # 출력 파일 경로
    output_path = "output_video.mp4"
    
    logger.info("=" * 60)
    logger.info("비디오 생성 시작")
    logger.info("=" * 60)
    logger.info(f"입력 텍스트: {len(english_text)}자")
    logger.info(f"출력 파일: {output_path}")
    if selected_channel:
        channels = list_available_channels()
        if selected_channel in channels:
            logger.info(f"채널: {channels[selected_channel].get('name', selected_channel)}")
    logger.info("=" * 60)
    
    try:
        # 파이프라인 초기화 (config.yaml 자동 로드, 채널 프로필 적용)
        config_path = Path("config.yaml")
        if config_path.exists():
            logger.info(f"설정 파일 사용: {config_path}")
        pipeline = VideoConversionPipeline(
            config_path=str(config_path) if config_path.exists() else None,
            channel=selected_channel
        )
        
        # 텍스트에서 비디오 생성
        result_path = pipeline.run_from_text(english_text, output_path, channel=selected_channel)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ 비디오 생성 완료!")
        logger.info(f"📁 파일 위치: {result_path}")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 중단되었습니다.")
        return 130
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

