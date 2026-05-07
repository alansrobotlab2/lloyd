#!/usr/bin/env python3
"""
Lloyd YouTube Capture Backend Server

Receives capture requests from Chrome extension, extracts transcript,
summarizes with local LLM, and saves to knowledge vault.
"""

import json
import os
import sys
import subprocess
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import re

# Configuration
HOST = 'localhost'
PORT = 8087  # Changed from 8080 to avoid conflicts
VAULT_PATH = os.path.expanduser('~/obsidian')
KNOWLEDGE_PATH = os.path.join(VAULT_PATH, 'knowledge', 'ai')
LOG_PATH = os.path.join(os.path.expanduser('~/lloyd/_pipeline'), 'youtube-capture.log')

# Ensure directories exist
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(KNOWLEDGE_PATH, exist_ok=True)

def log_message(message):
    """Append to log file"""
    timestamp = datetime.now().isoformat()
    with open(LOG_PATH, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:v=|youtu\.be/|/embed/|/v/)([A-Za-z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_transcript(video_id):
    """Fetch transcript using youtube-transcript-api via uv"""
    try:
        result = subprocess.run(
            [
                'uv', 'run', '--with', 'youtube-transcript-api',
                'python3', '-c',
                '''
from youtube_transcript_api import YouTubeTranscriptApi
import sys
video_id = sys.argv[1]
try:
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id)
    full_text = ' '.join(entry.text for entry in transcript)
    print(full_text)
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    sys.exit(1)
                ''',
                video_id
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return None, f"Transcript fetch failed: {result.stderr}"
        
        transcript_text = result.stdout.strip()
        if not transcript_text:
            return None, "Empty transcript"
        
        return transcript_text, None
        
    except subprocess.TimeoutExpired:
        return None, "Transcript fetch timed out"
    except Exception as e:
        return None, f"Transcript fetch error: {str(e)}"

def fetch_web_page(url):
    """Fetch and extract readable content from a web page"""
    try:
        # Use http_fetch tool logic via subprocess
        result = subprocess.run(
            [
                'uv', 'run', '--with', 'beautifulsoup4', '--with', 'requests',
                'python3', '-c',
                '''
import sys
import requests
from bs4 import BeautifulSoup

url = sys.argv[1]
try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove unwanted elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()
    
    # Get text and clean up
    text = soup.get_text(separator='\\n', strip=True)
    
    # Remove excessive whitespace
    lines = [line.strip() for line in text.split('\\n')]
    lines = [line for line in lines if line and not line.startswith('•')]
    clean_text = '\\n\\n'.join(lines)
    
    # Truncate if too long (keep ~15k chars)
    if len(clean_text) > 15000:
        clean_text = clean_text[:15000] + '\\n\\n... [truncated]'
    
    print(clean_text)
except Exception as e:
    print(f"ERROR:{e}", file=sys.stderr)
    sys.exit(1)
                ''',
                url
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return None, f"Page fetch failed: {result.stderr}"
        
        page_text = result.stdout.strip()
        if not page_text:
            return None, "Empty page content"
        
        return page_text, None
        
    except subprocess.TimeoutExpired:
        return None, "Page fetch timed out"
    except Exception as e:
        return None, f"Page fetch error: {str(e)}"

def summarize_with_llm(content, title, source_type, source_id=None):
    """Summarize content using local LLM"""
    try:
        # Use the available model from the running LLM
        available_model = "gemma-4-E4B-it-NVFP4"  # Default to available model
        
        # Prepare prompt based on content type
        if source_type == 'youtube':
            prompt = f"""Summarize this YouTube video transcript. Identify the core argument, key findings, and main takeaways. Be dense and specific — no filler.

VIDEO TITLE: {title}
VIDEO ID: {source_id}

TRANSCRIPT:
{content[:15000]}"""  # Truncate if too long (LLM context limit)
        else:
            prompt = f"""Summarize this web page article. Identify the core argument, key findings, and main takeaways. Be dense and specific — no filler.

PAGE TITLE: {title}
URL: {source_id}

CONTENT:
{content[:15000]}"""

        payload = json.dumps({
            "model": available_model,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_tokens": 1000,
            "temperature": 0.3
        }).encode()

        import urllib.request
        req = urllib.request.Request(
            "http://localhost:8091/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read())
            summary = result["choices"][0]["message"]["content"]
        
        return summary, None
        
    except Exception as e:
        return None, f"LLM summarization failed: {str(e)}"

def save_to_vault(source_id, title, url, content, summary, page_type):
    """Save knowledge note to vault"""
    try:
        # Generate slug from title
        slug = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')[:50]
        if not slug:
            slug = f"{page_type}-{source_id}"
        
        # Determine domain based on page type
        domain = "ai" if page_type == 'youtube' else "software"
        note_path = os.path.join(VAULT_PATH, 'knowledge', domain, f"{slug}.md")
        
        # Create YAML frontmatter
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        if page_type == 'youtube':
            frontmatter = f"""---
type: reference
tags: [youtube, transcript, summary, {domain}]
source: {url}
video_id: {source_id}
date: {date_str}
summary: "{summary[:150]}..."
---

# {title}

## Source
- **URL:** {url}
- **Video ID:** {source_id}
- **Captured:** {datetime.now().isoformat()}

## Summary
{summary}

## Full Transcript
{content}
"""
        else:
            frontmatter = f"""---
type: reference
tags: [article, web-page, summary, {domain}]
source: {url}
url: {url}
date: {date_str}
summary: "{summary[:150]}..."
---

# {title}

## Source
- **URL:** {url}
- **Captured:** {datetime.now().isoformat()}

## Summary
{summary}

## Full Content
{content}
"""
        
        # Write file
        with open(note_path, 'w') as f:
            f.write(frontmatter)
        
        return note_path, None
        
    except Exception as e:
        return None, f"Failed to save to vault: {str(e)}"

class CaptureHandler(BaseHTTPRequestHandler):
    """HTTP request handler for capture requests"""
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass
    
    def send_json_response(self, status_code, data):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_POST(self):
        """Handle capture POST requests"""
        if self.path != '/capture':
            self.send_json_response(404, {'error': 'Not found'})
            return
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())
            
            page_type = data.get('pageType', 'article')
            video_id = data.get('videoId')
            url = data.get('url')
            title = data.get('title', 'Unknown')
            
            if not url:
                self.send_json_response(400, {'error': 'Missing URL'})
                return
            
            log_message(f"Capture request for {page_type}: {url[:100]}")
            
            # Step 1: Fetch content based on type
            if page_type == 'youtube':
                if not video_id:
                    self.send_json_response(400, {'error': 'Missing videoId for YouTube'})
                    return
                
                content, error = fetch_transcript(video_id)
                source_id = video_id
            else:
                content, error = fetch_web_page(url)
                source_id = url
            
            if error:
                log_message(f"Content fetch failed: {error}")
                self.send_json_response(500, {'error': error})
                return
            
            log_message(f"Content fetched ({len(content)} chars)")
            
            # Step 2: Summarize
            summary, error = summarize_with_llm(content, title, page_type, source_id)
            if error:
                log_message(f"Summarization failed: {error}")
                self.send_json_response(500, {'error': error})
                return
            
            log_message(f"Summary generated ({len(summary)} chars)")
            
            # Step 3: Save to vault
            note_path, error = save_to_vault(source_id, title, url, content, summary, page_type)
            if error:
                log_message(f"Vault save failed: {error}")
                self.send_json_response(500, {'error': error})
                return
            
            log_message(f"Saved to {note_path}")
            
            self.send_json_response(200, {
                'success': True,
                'message': f'Captured {page_type} to {note_path}',
                'path': note_path,
                'summary': summary[:200] + '...' if len(summary) > 200 else summary
            })
            
        except json.JSONDecodeError:
            self.send_json_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            log_message(f"Unhandled error: {str(e)}")
            self.send_json_response(500, {'error': str(e)})

def main():
    """Start the server"""
    server = HTTPServer((HOST, PORT), CaptureHandler)
    log_message(f"Server starting on {HOST}:{PORT}")
    print(f"Lloyd YouTube Capture Backend running on http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_message("Server shutting down")
        print("\nShutting down...")
        server.shutdown()

if __name__ == '__main__':
    main()
