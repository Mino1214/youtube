"""SRT 자막 파일 생성 모듈"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """초를 SRT 타임스탬프 형식으로 변환 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_segment(segment_id: int, start_time: float, end_time: float, 
                      text: str) -> str:
    """단일 SRT 세그먼트 생성"""
    start_ts = format_timestamp(start_time)
    end_ts = format_timestamp(end_time)
    
    return f"{segment_id}\n{start_ts} --> {end_ts}\n{text}\n\n"


class SubtitleGenerator:
    """SRT 자막 생성기"""
    
    def __init__(self, mode: str = "korean_only", style: Optional[dict] = None):
        """
        Args:
            mode: "korean_only" 또는 "both_languages"
            style: 자막 스타일 설정
        """
        self.mode = mode
        self.style = style or {}
        logger.info(f"자막 생성기 초기화: 모드={mode}")
    
    def generate_srt(self, segments: List[Dict[str, Any]], 
                    output_path: Optional[str] = None) -> str:
        """
        세그먼트 리스트로부터 SRT 파일 생성
        
        Args:
            segments: 번역된 세그먼트 리스트
            output_path: 출력 SRT 파일 경로
            
        Returns:
            생성된 SRT 파일 경로
        """
        if output_path is None:
            output_path = "subtitles.srt"
        
        logger.info(f"SRT 자막 생성 중: {len(segments)}개 세그먼트")
        
        srt_content = []
        
        for i, segment in enumerate(segments, start=1):
            start_time = segment["start"]
            end_time = segment["end"]
            
            if self.mode == "korean_only":
                text = segment.get("text_ko", segment.get("text", ""))
            elif self.mode == "both_languages":
                text_en = segment.get("text_en", "")
                text_ko = segment.get("text_ko", segment.get("text", ""))
                text = f"{text_en}\n{text_ko}"
            else:
                text = segment.get("text", "")
            
            # 텍스트 정리
            text = text.strip().replace("\n", " ")
            
            if text:
                srt_segment = create_srt_segment(i, start_time, end_time, text)
                srt_content.append(srt_segment)
        
        # SRT 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(srt_content))
        
        logger.info(f"SRT 자막 생성 완료: {output_path}")
        return output_path
    
    def generate_ass(self, segments: List[Dict[str, Any]], 
                    output_path: Optional[str] = None) -> str:
        """
        ASS 자막 파일 생성 (FFmpeg 하드코딩에 더 적합)
        
        Args:
            segments: 번역된 세그먼트 리스트
            output_path: 출력 ASS 파일 경로
            
        Returns:
            생성된 ASS 파일 경로
        """
        if output_path is None:
            output_path = "subtitles.ass"
        
        logger.info(f"ASS 자막 생성 중: {len(segments)}개 세그먼트")
        
        # ASS 헤더
        font_name = self.style.get("font_name", "Arial")
        font_size = self.style.get("font_size", 24)
        primary_color = self.style.get("primary_color", "&H00FFFFFF")
        outline_color = self.style.get("outline_color", "&H00000000")
        outline_width = self.style.get("outline_width", 2)
        
        ass_content = [
            "[Script Info]",
            "Title: Korean Subtitles",
            "ScriptType: v4.00+",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{font_name},{font_size},{primary_color},&H00FFFFFF,{outline_color},&H00000000,0,0,0,0,100,100,0,0,1,{outline_width},0,2,10,10,10,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        
        # 이벤트 추가
        for segment in segments:
            start_time = format_timestamp(segment["start"]).replace(',', '.')
            end_time = format_timestamp(segment["end"]).replace(',', '.')
            
            if self.mode == "korean_only":
                text = segment.get("text_ko", segment.get("text", ""))
            elif self.mode == "both_languages":
                text_en = segment.get("text_en", "")
                text_ko = segment.get("text_ko", segment.get("text", ""))
                text = f"{text_en}\\N{text_ko}"
            else:
                text = segment.get("text", "")
            
            # ASS 형식 이스케이프
            text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            
            if text:
                ass_content.append(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}")
        
        # ASS 파일 저장
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write('\n'.join(ass_content))
        
        logger.info(f"ASS 자막 생성 완료: {output_path}")
        return output_path


def create_subtitle_generator(config: dict) -> SubtitleGenerator:
    """설정에서 SubtitleGenerator 인스턴스 생성"""
    subtitle_config = config.get("subtitles", {})
    return SubtitleGenerator(
        mode=subtitle_config.get("mode", "korean_only"),
        style=subtitle_config.get("style", {})
    )

