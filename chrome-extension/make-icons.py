#!/usr/bin/env python3
"""Generate simple PNG icons for the Chrome extension"""

import os

def create_icon(filename, size):
    """Create a simple blue circle icon"""
    try:
        from PIL import Image, ImageDraw
        
        img = Image.new('RGBA', (size, size), (26, 26, 46, 255))  # Dark background
        draw = ImageDraw.Draw(img)
        
        # Draw blue circle
        margin = 4
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=(0, 217, 255, 255)
        )
        
        # Draw "L" in center
        draw.text(
            (size // 2 - 6, size // 2 - 10),
            'L',
            fill=(26, 26, 46, 255),
            font_size=size // 2
        )
        
        img.save(filename)
        print(f"Created {filename}")
    except ImportError:
        print(f"PIL not available, creating placeholder {filename}")
        # Create a minimal valid PNG as placeholder
        with open(filename, 'wb') as f:
            # Minimal 1x1 blue pixel PNG
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82')

if __name__ == '__main__':
    icons = [
        ('icon16.png', 16),
        ('icon48.png', 48),
        ('icon128.png', 128)
    ]
    
    for filename, size in icons:
        create_icon(filename, size)
