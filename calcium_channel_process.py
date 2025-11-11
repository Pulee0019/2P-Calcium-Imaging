import tifffile
import numpy as np

def convert_tiff_sequence(input_path, output_path):
    with tifffile.TiffFile(input_path) as tif:
        frames = tif.asarray()
        total_frames = frames.shape[0]
        
        if total_frames % 2 != 0:
            raise ValueError("The total frames must be even number")
        
        mid_point = total_frames // 2
        
        new_sequence = np.empty_like(frames)
        
        for i in range(mid_point):
            new_sequence[2 * i] = frames[i]
            new_sequence[2 * i + 1] = frames[mid_point + i]
    
    tifffile.imwrite(output_path, new_sequence)

if __name__ == "__main__":
    input_file = r"D:\Expriment\Calcium\test1\Image_scan_1_region_0_0.tif"
    output_file = r"D:\Expriment\Calcium\test1\output\Image_scan_1_region_0_0_processed.tif"
    convert_tiff_sequence(input_file, output_file)