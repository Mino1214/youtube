"""오디오 추출 모듈 - FFmpeg 및 yt-dlp 통합"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple
import yt_dlp

logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    """입력 경로가 URL인지 확인"""
    return path.startswith(('http://', 'https://', 'www.'))


def download_video(url: str, output_path: Optional[str] = None) -> str:
    """yt-dlp를 사용하여 YouTube 또는 기타 URL에서 비디오 다운로드"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.mp4')
    
    logger.info(f"비디오 다운로드 중: {url}")
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path.replace('.mp4', '.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            # 실제 다운로드된 파일 경로 찾기
            info = ydl.extract_info(url, download=False)
            ext = info.get('ext', 'mp4')
            downloaded_path = output_path.replace('.mp4', f'.{ext}')
            if os.path.exists(downloaded_path):
                return downloaded_path
            return output_path
    except Exception as e:
        logger.error(f"비디오 다운로드 실패: {e}")
        raise


def extract_audio(video_path: str, output_path: Optional[str] = None, 
                  sample_rate: int = 16000) -> str:
    """FFmpeg를 사용하여 비디오에서 오디오 추출"""
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.wav')
    
    logger.info(f"오디오 추출 중: {video_path}")
    
    # FFmpeg 명령어 구성
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ar', str(sample_rate),  # 샘플 레이트
        '-ac', '1',  # 모노 채널
        '-y',  # 덮어쓰기
        output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"오디오 추출 완료: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"오디오 추출 실패: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("FFmpeg를 찾을 수 없습니다. FFmpeg가 설치되어 있는지 확인하세요.")
        raise


def get_video_info(video_path: str) -> dict:
    """비디오 파일 정보 추출"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        import json
        return json.loads(result.stdout)
    except Exception as e:
        logger.warning(f"비디오 정보 추출 실패: {e}")
        return {}


def process_input(input_path: str, temp_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    입력 경로 처리 (URL 또는 로컬 파일)
    
    Returns:
        Tuple[비디오_경로, 오디오_경로]
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # URL인 경우 다운로드
    if is_url(input_path):
        video_path = download_video(input_path, str(temp_dir / "downloaded_video.mp4"))
    else:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {input_path}")
        video_path = input_path
    
    # 오디오 추출
    audio_path = str(temp_dir / "extracted_audio.wav")
    audio_path = extract_audio(video_path, audio_path)
    
    return video_path, audio_path

