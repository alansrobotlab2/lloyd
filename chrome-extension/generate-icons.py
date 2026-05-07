#!/usr/bin/env python3
"""Generate simple PNG icons for the Chrome extension using only standard library"""

import struct
import zlib

def create_png(filename, size, color=(0, 217, 255)):
    """Create a simple solid-color PNG icon"""
    
    # PNG signature
    signature = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk
    ihdr_data = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    
    # IDAT chunk (image data)
    raw_data = b''
    for y in range(size):
        raw_data += b'\x00'  # Filter byte
        for x in range(size):
            # Solid color
            raw_data += bytes(color)
    
    compressed = zlib.compress(raw_data, 9)
    idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
    
    # IEND chunk
    iend_crc = zlib.crc32(b'IEND') & 0xffffffff
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    
    # Write file
    with open(filename, 'wb') as f:
        f.write(signature + ihdr + idat + iend)
    
    print(f"Created {filename} ({size}x{size})")

if __name__ == '__main__':
    sizes = [16, 48, 128]
    color = (0, 217, 255)  # Cyan/blue
    
    for size in sizes:
        filename = f'icon{size}.png'
        create_png(filename, size, color)
    
    print("\nIcons generated successfully!")
