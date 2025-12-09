"""FFmpeg를 사용한 비디오 렌더링 모듈"""

import logging
import subprocess
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class VideoRenderer:
    """비디오 렌더러 - 오디오, 자막 결합"""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: 비디오 설정 딕셔너리
        """
        self.config = config or {}
        video_config = self.config.get("video", {})
        self.video_codec = video_config.get("video_codec", "libx264")
        self.audio_codec = video_config.get("audio_codec", "aac")
        self.video_bitrate = video_config.get("video_bitrate", "5000k")
        self.audio_bitrate = video_config.get("audio_bitrate", "192k")
        self.preserve_fps = video_config.get("preserve_fps", True)
        
        logger.info("비디오 렌더러 초기화")
    
    def render(self, video_path: str, audio_path: str, subtitle_path: str,
              output_path: str) -> str:
        """
        비디오, 오디오, 자막을 결합하여 최종 비디오 생성
        
        Args:
            video_path: 원본 비디오 경로 (음소거됨)
            audio_path: 한국어 나레이션 오디오 경로
            subtitle_path: SRT 또는 ASS 자막 파일 경로
            output_path: 출력 비디오 경로
            
        Returns:
            생성된 비디오 파일 경로
        """
        logger.info("비디오 렌더링 시작...")
        
        # 자막 파일 형식 확인
        subtitle_ext = Path(subtitle_path).suffix.lower()
        use_ass = subtitle_ext == '.ass'
        
        # FFmpeg 명령어 구성
        cmd = ['ffmpeg']
        
        # 입력 파일 추가
        cmd.extend(['-i', video_path])  # 비디오 입력
        cmd.extend(['-i', audio_path])  # 오디오 입력
        
        # 비디오 스트림 선택 및 음소거
        cmd.extend(['-map', '0:v:0'])  # 첫 번째 입력의 비디오 스트림
        cmd.extend(['-map', '1:a:0'])  # 두 번째 입력의 오디오 스트림
        
        # 비디오 코덱 설정
        cmd.extend(['-c:v', self.video_codec])
        cmd.extend(['-b:v', self.video_bitrate])
        
        # 프레임레이트 유지
        if self.preserve_fps:
            cmd.extend(['-r', '30'])  # 기본값, 실제로는 원본에서 추출해야 함
        
        # 오디오 코덱 설정
        cmd.extend(['-c:a', self.audio_codec])
        cmd.extend(['-b:a', self.audio_bitrate])
        
        # 자막 필터 추가
        if use_ass:
            # ASS 자막은 필터로 추가
            cmd.extend([
                '-vf', f"subtitles='{subtitle_path}':force_style='FontName=Arial,FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2'"
            ])
        else:
            # SRT 자막도 필터로 추가
            cmd.extend([
                '-vf', f"subtitles='{subtitle_path}':force_style='FontName=Arial,FontSize=24,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2'"
            ])
        
        # 오디오 동기화 (필요시)
        cmd.extend(['-shortest'])  # 가장 짧은 스트림에 맞춤
        
        # 출력 파일
        cmd.extend(['-y', output_path])
        
        try:
            logger.info("FFmpeg 실행 중...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"비디오 렌더링 완료: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"비디오 렌더링 실패: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("FFmpeg를 찾을 수 없습니다. FFmpeg가 설치되어 있는지 확인하세요.")
            raise
    
    def mute_original_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        원본 비디오의 오디오를 음소거
        
        Args:
            video_path: 원본 비디오 경로
            output_path: 출력 비디오 경로
            
        Returns:
            음소거된 비디오 파일 경로
        """
        if output_path is None:
            output_path = video_path.replace('.mp4', '_muted.mp4')
        
        logger.info(f"원본 오디오 음소거 중: {video_path}")
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-c', 'copy',
            '-an',  # 오디오 제거
            '-y',
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"오디오 음소거 완료: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"오디오 음소거 실패: {e.stderr}")
            raise
    
    def get_video_duration(self, video_path: str) -> float:
        """비디오 길이(초) 반환"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"비디오 길이 추출 실패: {e}")
            return 0.0
    
    def get_video_fps(self, video_path: str) -> float:
        """비디오 프레임레이트 반환"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                return num / den
            return float(fps_str)
        except Exception as e:
            logger.warning(f"프레임레이트 추출 실패: {e}")
            return 30.0  # 기본값


def create_video_renderer(config: dict) -> VideoRenderer:
    """설정에서 VideoRenderer 인스턴스 생성"""
    return VideoRenderer(config)

