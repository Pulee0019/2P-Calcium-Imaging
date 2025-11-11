import tifffile
import numpy as np
import os
from pathlib import Path

def remove_yellow_channel(input_path, output_red_path, output_green_path):
    os.makedirs(os.path.dirname(output_red_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_green_path), exist_ok=True)
    
    with tifffile.TiffFile(input_path) as tif:
        frames = tif.asarray()
        total_frames = frames.shape[0]
        
        if total_frames % 3 != 0:
            raise ValueError("The total frames must be divisible by 3")
        
        frames_per_channel = total_frames // 3
        
        green_frames = frames[:frames_per_channel]
        red_frames = frames[frames_per_channel:2*frames_per_channel]
        
        tifffile.imwrite(output_green_path, green_frames)
        tifffile.imwrite(output_red_path, red_frames)

if __name__ == "__main__":
    input_file = r"F:\PCW's program\calcium imaging\calcium data\20250910\b\average 2_001\Image_scan_1_region_0_0.tif"
    
    output_red = r"F:\PCW's program\calcium imaging\calcium data\20250910\b\output\red\Image_scan_1_region_0_0_red.tif"
    output_green = r"F:\PCW's program\calcium imaging\calcium data\20250910\b\output\green\Image_scan_1_region_0_0_green.tif"
    
    remove_yellow_channel(input_file, output_red, output_green)