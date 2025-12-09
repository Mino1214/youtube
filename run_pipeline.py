#!/usr/bin/env python3
"""로컬 AI 자동 비디오 변환 파이프라인 CLI 진입점"""

import argparse
import logging
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline import VideoConversionPipeline

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='로컬 AI 자동 비디오 변환: 외국 비디오를 한국어 나레이션과 자막이 있는 비디오로 변환',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 로컬 파일 변환
  python run_pipeline.py --input video.mp4 --output output.mp4
  
  # YouTube URL 변환
  python run_pipeline.py --input "https://www.youtube.com/watch?v=..." --output output.mp4
  
  # 설정 파일 지정
  python run_pipeline.py --input video.mp4 --output output.mp4 --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        required=True,
        help='입력 비디오 파일 경로 또는 URL (YouTube 등)'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help='출력 비디오 파일 경로'
    )
    
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        default='config.yaml',
        help='설정 파일 경로 (기본값: config.yaml)'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='상세 로그 출력 (DEBUG 레벨)'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='작업 완료 후 임시 파일 유지 (디버깅용)'
    )
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 입력 파일/URL 확인
    if not args.input.startswith(('http://', 'https://', 'www.')):
        if not os.path.exists(args.input):
            logger.error(f"입력 파일을 찾을 수 없습니다: {args.input}")
            sys.exit(1)
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"출력 디렉토리 생성: {output_dir}")
    
    # 설정 파일 확인
    config_path = args.config
    if not os.path.exists(config_path):
        logger.warning(f"설정 파일을 찾을 수 없습니다: {config_path}")
        logger.info("기본 설정을 사용합니다.")
        config_path = None
    
    try:
        # 파이프라인 실행
        pipeline = VideoConversionPipeline(
            config_path=config_path
        )
        
        # 자동 정리 설정
        if args.no_cleanup:
            pipeline.auto_cleanup = False
            logger.info("임시 파일 정리 비활성화됨")
        
        # 파이프라인 실행
        output_file = pipeline.run(args.input, args.output)
        
        logger.info(f"\n✅ 변환 완료!")
        logger.info(f"출력 파일: {output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n사용자에 의해 중단되었습니다.")
        return 130
    except Exception as e:
        logger.error(f"\n❌ 오류 발생: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())

