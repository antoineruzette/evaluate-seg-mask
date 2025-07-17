#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image

def convert_rgb_to_rg_tiff(input_path: str, output_path: str) -> None:
    """
    Convert an image to a 2-channel TIFF. Handles both:
    - RGB images (takes first two channels)
    - 16-bit single channel images (splits into high/low bytes)
    Saves in CYX order (Channels, Height, Width) for direct channel access.
    
    Args:
        input_path: Path to the input PNG file
        output_path: Path where to save the output TIFF file
    """
    # Read the input image using PIL
    img = Image.open(input_path)
    img_array = np.array(img)
    
    # Handle different input types
    if img_array.dtype == np.uint16:
        # Split 16-bit values into two 8-bit channels
        high_byte = (img_array >> 8).astype(np.uint8)
        low_byte = (img_array & 0xFF).astype(np.uint8)
        # Stack in CYX order
        two_channel = np.stack([high_byte, low_byte], axis=0)
    else:
        # For RGB or other format images, take first two channels
        if len(img_array.shape) < 3:
            raise ValueError(f"Input image {input_path} must be either 16-bit or have multiple channels")
        # Convert to CYX order directly
        two_channel = np.moveaxis(img_array[:, :, :2], 2, 0)
    
    # Save with tifffile to preserve dimension order
    tifffile.imwrite(output_path, two_channel, photometric='minisblack')
    print(f"Converted {input_path} to {output_path}")

def process_directory(input_dir: Path, output_dir: Path) -> None:
    """
    Process all PNG files in input directory and save RG TIFF versions to output directory.
    
    Args:
        input_dir: Path to input directory containing PNG files
        output_dir: Path where to save output TIFF files
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG files in input directory
    png_files = list(input_dir.glob("**/*.png"))
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(png_files)} PNG files to process")
    
    for png_file in png_files:
        # Create corresponding output path
        relative_path = png_file.relative_to(input_dir)
        output_path = output_dir / relative_path.with_suffix('.tiff')
        
        # Create parent directories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            convert_rgb_to_rg_tiff(str(png_file), str(output_path))
        except Exception as e:
            print(f"Error processing {png_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Convert RGB PNG to RG TIFF')
    parser.add_argument('input', type=str, help='Input PNG file or directory path')
    parser.add_argument('output', type=str, help='Output TIFF file or directory path')
    
    args = parser.parse_args()
    
    # Convert paths to absolute paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    # Check if input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    if input_path.is_dir():
        # Process directory
        process_directory(input_path, output_path)
    else:
        # Process single file
        if not input_path.suffix.lower() == '.png':
            raise ValueError("Input file must be a PNG file")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        convert_rgb_to_rg_tiff(str(input_path), str(output_path))

if __name__ == '__main__':
    main() 