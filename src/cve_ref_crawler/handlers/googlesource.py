# src/cve_ref_crawler/handlers/googlesource.py

import base64
from urllib.parse import urlparse
from typing import Optional

def is_googlesource_url(url: str) -> bool:
    """Check if URL is from android.googlesource.com"""
    # Remove any trailing parenthesis that might have been captured
    url = url.rstrip(')')
    parsed = urlparse(url)
    return parsed.netloc == 'android.googlesource.com'

def handle_googlesource_url(url: str) -> Optional[str]:
    """
    Handle Android Git Source URLs
    Returns modified URL for fetching content
    """
    # Clean up URL first
    url = url.rstrip(')')  # Remove any trailing parenthesis
    
    # Convert %2B to + if present
    if '%2B' in url:
        url = url.replace('%2B', '+')
    
    if '/+/' in url and '?' not in url:
        return f"{url}?format=TEXT"
        
    return url

def parse_googlesource_response(response_text: str) -> Optional[str]:
    """
    Parse Google Source response
    Returns decoded content or None if parsing fails
    """
    try:
        # Content is base64 encoded
        decoded = base64.b64decode(response_text).decode('utf-8')
        
        # Parse commit info and message
        lines = decoded.split('\n')
        commit_info = {}
        message_lines = []
        in_message = False
        
        for line in lines:
            if not line.strip():
                in_message = True
                continue
                
            if not in_message:
                if ' ' in line:
                    key, value = line.split(' ', 1)
                    commit_info[key] = value
            else:
                message_lines.append(line)
        
        # Format the output
        formatted = f"""
            Commit Information:
            ------------------
            Author: {commit_info.get('author', 'Unknown')}
            Date: {commit_info.get('committer', 'Unknown')}
            Bug ID: {' '.join(line for line in message_lines if line.startswith('Bug:'))}

            Commit Message:
            --------------
            {''.join(message_lines)}
            """
        return formatted
                    
    except Exception as e:
        print(f"Error decoding googlesource response: {e}")
        return None