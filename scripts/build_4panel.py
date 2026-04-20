#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"

# Note: The first image is the Stage IV NWPS map, the other 3 are the new station maps
PANELS = [
    DOCS_DIR / "lix_24h_precip_latest.png",
    DOCS_DIR / "lix_station_air_temp_daily_max_latest.png",
    DOCS_DIR / "lix_station_air_temp_daily_min_latest.png",
    DOCS_DIR / "lix_station_air_temp_latest.png"
]

def main():
    # Verify all files exist before trying to open them
    for panel in PANELS:
        if not panel.exists():
            print(f"Warning: {panel.name} is missing. The 4-panel cannot be generated until all images exist.")
            return

    # Open all 4 images
    images = [Image.open(img) for img in PANELS]
    
    # We assume all images are the standard 16x11.5 aspect ratio created by matplotlib
    w, h = images[0].size
    
    # Grid math: 2 columns, 2 rows, plus a 100-pixel white header block at the top
    header_height = 100
    new_w = w * 2
    new_h = (h * 2) + header_height
    
    # Create the white background canvas
    canvas = Image.new('RGB', (new_w, new_h), 'white')
    
    # Paste images into the 4 quadrants
    canvas.paste(images[0], (0, header_height))         # Top Left: Precip
    canvas.paste(images[1], (w, header_height))         # Top Right: Max Temp
    canvas.paste(images[2], (0, h + header_height))     # Bottom Left: Min Temp
    canvas.paste(images[3], (w, h + header_height))     # Bottom Right: Current Temp
    
    # Draw the main title and grid borders
    draw = ImageDraw.Draw(canvas)
    
    # Horizontal line separating header from maps
    draw.line([(0, header_height), (new_w, header_height)], fill="black", width=5)
    # Vertical line down the middle
    draw.line([(w, header_height), (w, new_h)], fill="black", width=5)
    # Horizontal line across the middle
    draw.line([(0, h + header_height), (new_w, h + header_height)], fill="black", width=5)
    
    out_path = DOCS_DIR / "lix_4panel_latest.png"
    canvas.save(out_path)
    print(f"Successfully stitched 4-panel image to {out_path.name}")

if __name__ == "__main__":
    main()
