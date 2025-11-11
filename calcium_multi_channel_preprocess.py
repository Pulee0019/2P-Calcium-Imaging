import os
import tifffile
import numpy as np
import tkinter as tk
from tkinter import filedialog

def convert_tiff_sequence(input_file, output_dir):
    with tifffile.TiffFile(input_file) as tif:
        frames = tif.asarray()
        total_frames = frames.shape[0]
        
        if total_frames % 2 != 0:
            raise ValueError("The total frames must be even number")
        
        mid_point = total_frames // 2
        
        new_sequence = np.empty_like(frames)
        
        for i in range(mid_point):
            new_sequence[2 * i] = frames[i]
            new_sequence[2 * i + 1] = frames[mid_point + i]
    
    file_name = input_file.split("/")[-1].split(".")[0] + "_processed.tif"
    output_file = os.path.join(output_dir, file_name)
    if not os.path.exists(output_file):
        tifffile.imwrite(output_file, new_sequence)
        print(f"Process completed! result save as {output_file}.")
    else:
        print(f"{output_file} already existed!")

if __name__ == "__main__":
    input_file = filedialog.askopenfilename(
        title="Please select TIFF file",
        filetypes=[("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")]
    )

    if not input_file:
        print("No file selected, program terminated.")
        exit()

    base_dir = os.path.dirname(os.path.dirname(input_file))
    output_dir = os.path.join(base_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"{output_dir} already existed!")

    convert_tiff_sequence(input_file, output_dir)
    
    ch1_dir = os.path.join(output_dir, "ch1")
    ch2_dir = os.path.join(output_dir, "ch2")
    if not os.path.exists(ch1_dir):
        os.makedirs(ch1_dir)
    else:
        print(f"{ch1_dir} already existed!")
    
    if not os.path.exists(ch2_dir):
        os.makedirs(ch2_dir)
    else:
        print(f"{ch2_dir} already existed!")

    