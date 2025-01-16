# src/cve_ref_crawler/llm_extractor.py
import os
import re
from dotenv import load_dotenv, dotenv_values
import google.generativeai as genai
from pathlib import Path
from typing import Optional
import logging
import time
from collections import defaultdict
from .utils.logging_utils import setup_logging
from config import LOG_CONFIG, LLM_SETTINGS

# Load environment variables
load_dotenv()
config = dotenv_values("../../env/.env")

# Configure API key
os.environ["GEMINI_API_KEY"] = config['GOOGLE_API_KEY']

# Define safety settings
safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class VulnerabilityExtractor:
    def __init__(self, output_dir: str):
        """Initialize the vulnerability extractor"""
        self.output_dir = Path(output_dir)
        self.logger = setup_logging(
            log_dir=LOG_CONFIG["dir"],
            log_level=LOG_CONFIG["level"],
            module_name=__name__
        )
        
        # Initialize Gemini
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        
        # Configure generation parameters
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        # Initialize model with custom prompt template
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
            system_instruction=self._get_system_prompt(),
            safety_settings=safe
        )

    def __init__(self, output_dir: str):
        """Initialize the vulnerability extractor"""
        self.output_dir = Path(output_dir)
        self.logger = setup_logging(
            log_dir=LOG_CONFIG["dir"],
            log_level=LOG_CONFIG["level"],
            module_name=__name__
        )
        
        # Initialize Gemini
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        
        # Rate limiting state
        self.last_request_time = time.time()
        self.backoff_time = LLM_SETTINGS["backoff"]["initial"]
        self.request_times = []  # Track request timestamps
        
        # Configure generation parameters from config
        self.generation_config = {
            "temperature": LLM_SETTINGS["model"]["temperature"],
            "top_p": LLM_SETTINGS["model"]["top_p"],
            "top_k": LLM_SETTINGS["model"]["top_k"],
            "max_output_tokens": LLM_SETTINGS["model"]["max_output_tokens"],
            "response_mime_type": LLM_SETTINGS["model"]["response_mime_type"],
        }
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=LLM_SETTINGS["model"]["name"],
            generation_config=self.generation_config,
            system_instruction=self._get_system_prompt(),
            safety_settings=safe
        )

    def _wait_for_rate_limit(self):
        """Implement rate limiting with exponential backoff"""
        current_time = time.time()
        settings = LLM_SETTINGS["rate_limits"]
        
        # Clean up old request times
        self.request_times = [t for t in self.request_times 
                            if current_time - t < settings["requests_window"]]
        
        # Check if we're over the rate limit
        if len(self.request_times) >= settings["max_requests_per_window"]:
            sleep_time = max(
                0,
                settings["requests_window"] - (current_time - self.request_times[0])
            )
            self.logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
            self.request_times = self.request_times[1:]
        
        # Ensure minimum time between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < settings["min_request_interval"]:
            time.sleep(settings["min_request_interval"] - time_since_last)

        self.last_request_time = time.time()
        self.request_times.append(self.last_request_time)

    def _handle_api_error(self, e: Exception, cve_id: str) -> None:
        """Handle API errors with exponential backoff"""
        if "429" in str(e):  # Rate limit exceeded
            sleep_time = min(self.backoff_time, LLM_SETTINGS["backoff"]["max"])
            self.logger.warning(
                f"Rate limit exceeded for {cve_id}. "
                f"Backing off for {sleep_time:.1f} seconds"
            )
            time.sleep(sleep_time)
            self.backoff_time *= 2  # Exponential backoff
        else:
            self.logger.error(f"API error for {cve_id}: {str(e)}")

    def _get_system_prompt(self) -> str:
        """Returns the system prompt template for vulnerability extraction"""
        return """Guidelines:
1. First verify if the content relates to the CVE specified based on the official description
2. If the content does not relate to this CVE, respond with "UNRELATED"
3. If no useful vulnerability information is found, respond with "NOINFO" 
4. For relevant content, extract:
   - Root cause of vulnerability
   - Weaknesses/vulnerabilities present
   - Impact of exploitation
   - Attack vectors
   - Required attacker capabilities/position

Additional instructions:
- Preserve original technical details and descriptions
- Remove unrelated content
- Translate non-English content to English
- Note if the content provides more detail than the official CVE description
"""

    def _get_user_prompt(self, cve_id: str, cve_desc: str, content: str) -> str:
        """Formats the user prompt with the specific CVE information and content"""
        return f"""(CVE) ID: {cve_id}
CVE Description: {cve_desc}

===CONTENT to ANALYZE===

{content}"""

    def is_cve_extracted(self, cve_id: str) -> bool:
        """
        Check if vulnerability info has already been extracted by checking for refined.md
        
        Args:
            cve_id: CVE ID to check
            
        Returns:
            bool: True if refined.md exists
        """
        refined_file = self.output_dir / cve_id / "refined" / "refined.md"
        if refined_file.exists():
            self.logger.info(f"Skipping {cve_id} - refined.md already exists")
            return True
        return False
    
    def _parse_filename(self, filename: str) -> Optional[dict]:
        """Parse filename into components."""
        pattern = r'^(.+?)_([a-f0-9]+)_(\d{8}_\d{6})\.html$'
        match = re.match(pattern, filename)
        if match:
            domain, hash_val, datetime_str = match.groups()
            return {
                'domain': domain,
                'hash': hash_val,
                'datetime': datetime_str,
                'full_name': filename
            }
        return None

    def _should_delete(self, file_info: dict, latest_info: dict, base_path: Path) -> bool:
        """Check if file should be deleted based on size comparison."""
        if not latest_info:
            return False
            
        current_path = base_path / file_info['full_name']
        latest_path = base_path / latest_info['full_name']
        
        # Compare file sizes
        if current_path.stat().st_size != latest_path.stat().st_size:
            return False
            
        return True

    def _cleanup_directory(self, directory_path: Path) -> tuple[int, int]:
        """Remove duplicate files keeping only the latest version."""
        # Group files by domain and hash
        file_groups = defaultdict(list)
        
        # Process all HTML files
        for file_path in directory_path.glob('*.html'):
            file_info = self._parse_filename(file_path.name)
            if file_info:
                key = (file_info['domain'], file_info['hash'])
                file_groups[key].append(file_info)
        
        deleted_count = 0
        total_count = 0
        
        # Process each group
        for key, files in file_groups.items():
            total_count += len(files)
            if len(files) == 1:
                continue
                
            # Sort files by datetime string (newest first)
            files.sort(key=lambda x: x['datetime'], reverse=True)
            latest = files[0]
            
            # Check each older file
            for file_info in files[1:]:
                if self._should_delete(file_info, latest, directory_path):
                    file_path = directory_path / file_info['full_name']
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        self.logger.info(f"Deleted duplicate: {file_info['full_name']}")
                    except OSError as e:
                        self.logger.error(f"Error deleting {file_path}: {e}")
        
        return total_count, deleted_count
    
    def _is_text_file(self, file_path: Path) -> bool:
        """
        Check if a file contains text content by:
        1. Checking the file extension
        2. Attempting to decode content as text
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if file contains valid text content
        """
        # First check extension
        text_extensions = {'.txt', '.html', '.htm', '.md', '.csv', '.log'}
        if file_path.suffix.lower() not in text_extensions:
            return False
            
        # Try reading the first few KB as text
        try:
            with open(file_path, 'rb') as f:
                # Read first 8KB
                sample = f.read(8192)
                
            # Try decoding as UTF-8
            try:
                sample.decode('utf-8')
                return True
            except UnicodeDecodeError:
                pass
                
            # Try decoding as ASCII
            try:
                sample.decode('ascii')
                return True
            except UnicodeDecodeError:
                pass
                
            # Additional text encodings could be added here
            
            return False
    
        except Exception as e:
            self.logger.error(f"Error checking if {file_path} is text: {e}")
            return False
    
    def _get_text_content(self, file_path: Path) -> Optional[str]:
        """
        Read text content from file, handling different encodings.
        
        Args:
            file_path: Path to text file
            
        Returns:
            str: File content if successful, None if failed
        """
        encodings = ['utf-8', 'ascii', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error reading {file_path} with {encoding}: {e}")
                return None
                
        self.logger.warning(f"Could not decode {file_path} with any supported encoding")
        return None
    
    def process_cve(self, cve_id: str) -> bool:
        """Process all text files for a CVE and extract vulnerability information"""
        # Check if already processed
        if self.is_cve_extracted(cve_id):
            return True
            
        # Setup directories
        text_dir = self.output_dir / cve_id / "text"
        refined_dir = self.output_dir / cve_id / "refined"
            
        if not text_dir.exists():
            self.logger.warning(f"No text directory found for {cve_id}")
            return False
            
        try:
            # Create refined directory
            refined_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean up duplicate files first
            total_files, deleted_files = self._cleanup_directory(text_dir)
            self.logger.info(
                f"Cleaned up duplicates for {cve_id}: "
                f"removed {deleted_files} duplicates out of {total_files} files"
            )
            
            # Combine all text content
            all_content = []
            skipped_files = []
            text_files = list(text_dir.iterdir())
            self.logger.debug(f"Found {len(text_files)} files in {text_dir}")
            
            for text_file in text_files:
                if not text_file.is_file():  # Skip directories
                    continue
                    
                # Check if file contains text content
                if not self._is_text_file(text_file):
                    skipped_files.append(text_file.name)
                    continue
                    
                # Try to read text content
                content = self._get_text_content(text_file)
                if content is not None:
                    all_content.append(f"=== Content from {text_file.name} ===\n{content}\n")
                    self.logger.debug(f"Successfully read content from {text_file.name}")
                else:
                    skipped_files.append(text_file.name)
            
            if skipped_files:
                self.logger.warning(
                    f"Skipped {len(skipped_files)} non-text files in {cve_id}: "
                    f"{', '.join(skipped_files[:5])}{'...' if len(skipped_files) > 5 else ''}"
                )
            
            if not all_content:
                self.logger.warning(f"No valid text content found for {cve_id}")
                return False
                
            # Save combined raw content
            with open(refined_dir / "combined.md", 'w', encoding='utf-8') as f:
                f.write("\n".join(all_content))
                
            # Extract vulnerability information using LLM
            extracted_info = self._extract_vulnerability_info(
                cve_id=cve_id,
                content="\n".join(all_content)
            )
            
            if extracted_info == "NOINFO":
                self.logger.info(f"No vulnerability information found for {cve_id}")
                return False
                
            # Save extracted information
            with open(refined_dir / "refined.md", 'w', encoding='utf-8') as f:
                f.write(extracted_info)
                
            self.logger.info(f"Successfully processed {cve_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {cve_id}: {e}")
            return False

    def _extract_vulnerability_info(self, cve_id: str, content: str) -> str:
        """Extract vulnerability information using LLM with rate limiting"""
        try:
            cve_desc = "PLACEHOLDER - Implement CVE description retrieval"
            prompt = self._get_user_prompt(cve_id, cve_desc, content)
            
            for attempt in range(LLM_SETTINGS["retries"]["max_attempts"]):
                try:
                    # Apply rate limiting
                    self._wait_for_rate_limit()
                    
                    # Start chat session
                    chat = self.model.start_chat()
                    response = chat.send_message(prompt)
                    
                    # Reset backoff on success
                    self.backoff_time = LLM_SETTINGS["backoff"]["initial"]
                    
                    if not response.text:
                        self.logger.warning(f"Empty response from LLM for {cve_id}")
                        continue
                    
                    result = response.text.strip()
                    if result in ["UNRELATED", "NOINFO"]:
                        return result
                        
                    return result
                    
                except Exception as e:
                    self._handle_api_error(e, cve_id)
                    if attempt == LLM_SETTINGS["retries"]["max_attempts"] - 1:
                        raise
                    continue
            
            self.logger.error(f"All {LLM_SETTINGS['retries']['max_attempts']} attempts failed for {cve_id}")
            return "NOINFO"
            
        except Exception as e:
            self.logger.error(f"Error extracting vulnerability info for {cve_id}: {e}")
            return "NOINFO"