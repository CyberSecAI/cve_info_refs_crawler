# src/cve_ref_crawler/handlers/youtube.py

import re
from urllib.parse import urlparse, parse_qs
from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import logging

logger = logging.getLogger(__name__)

def is_youtube_url(url: str) -> bool:
    """Check if URL is from YouTube"""
    parsed = urlparse(url)
    return parsed.netloc in ('www.youtube.com', 'youtube.com', 'youtu.be')

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL
    """
    try:
        # Handle youtu.be URLs
        if 'youtu.be' in url:
            return url.split('/')[-1].split('?')[0]
            
        # Handle youtube.com URLs
        parsed = urlparse(url)
        if parsed.hostname in ('www.youtube.com', 'youtube.com'):
            if 'watch' in url:
                # Regular watch URL
                query = parse_qs(parsed.query)
                return query.get('v', [None])[0]
            else:
                # Handle /v/ or /embed/ URLs
                path = parsed.path
                if '/v/' in path or '/embed/' in path:
                    return path.split('/')[-1]
        
        return None
    except Exception as e:
        logger.error(f"Error extracting video ID from {url}: {e}")
        return None

def get_youtube_transcript(video_id: str) -> Optional[str]:
    """
    Get transcript for YouTube video
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        if not transcript:
            logger.warning(f"No transcript found for video {video_id}")
            return None
            
        # Format transcript entries with timestamps
        formatted_entries = []
        for entry in transcript:
            timestamp = f"{int(entry['start'] // 60):02d}:{int(entry['start'] % 60):02d}"
            formatted_entries.append(f"[{timestamp}] {entry['text']}")
            
        return '\n'.join(formatted_entries)
        
    except Exception as e:
        logger.error(f"Error getting YouTube transcript for {video_id}: {e}")
        return None

def handle_youtube_url(url: str) -> Optional[str]:
    """
    Extract transcript from YouTube URL
    """
    try:
        video_id = extract_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            return None
            
        transcript = get_youtube_transcript(video_id)
        if transcript:
            logger.info(f"Successfully extracted transcript for video {video_id}")
            return transcript
        else:
            logger.warning(f"No transcript available for video {video_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error handling YouTube URL {url}: {e}")
        return None