# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:55:31 2025

@author: Pulee
"""

import os
import shutil
import suite2p
import tifffile
import numpy as np
import tkinter as tk
from tkinter import filedialog
from suite2p.run_s2p import run_s2p

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
    return output_file

def run_suite2p_channel(data_path, channel_dir, functional_chan):
    ops = suite2p.default_ops()
    
    ops.update({
        # 'file_type': 'tif',
        'data_path': [data_path],

        'save_path0': channel_dir,

        ######## Main settings ########
        # 'nplanes': 1,
        'nchannels': 2,
        'functional_chan': functional_chan,
        'tau': 1.5,
        'fs': 15.136,
        # 'do_bidiphase': False,
        # 'multiplane_parallel': False,
        # 'ignore_flyback': -1,
        
        ######## Output settings ########
        # 'preclassify': 0.25,
        'save_NWB': True,
        'save_mat': True,
        # 'combined': True,
        # 'reg_tif': False,
        # 'reg_tif_chan2': False,
        # 'aspect': 1.0,
        # 'delete_bin': False,
        # 'move_bin': False,

        ######## Registration ########
        # 'do_registration': True,
        'align_by_chan': 1,
        # 'nimg_init': 300,
        # 'batch_size': 500,
        # 'smooth_sigma': 1.15,
        # 'smooth_sigma_time': 0,
        # 'maxregshift': 0.1,
        # 'th_badframes': 1.0,
        # 'keep_movie_raw': False,
        # 'two_step_registration': False,

        ######## Nongrid ########
        # 'nonrigid': True,
        # 'block_size': [128, 128],
        # 'snr_thresh': 1.2,
        # 'maxregshiftNR': 5,

        ######## 1P ########
        # '1Preg': 0,
        # 'spatial_hp_reg': 42,
        # 'pre_smooth': False,
        # 'spatial_taper': 40,
        
        ######## Functional detect ########
        # 'roidetect': True,
        # 'sparse_mode': True,
        # 'denoise': True,
        # 'spatial_scale': 2,
        # 'connected': True,
        # 'threshold_scaling': 1.0,
        # 'max_overlap': 0.75,
        # 'max_iterations': 200,
        # 'high_pass': 100,
        # 'spatial_hp_detect': 25,

        ######## Anat detect ########
        'anatomical_only': 3,
        # 'diameter': 0,
        # 'cellprob_threshold': 0.0,
        # 'flow_threshold': 1.2,
        # 'pretrained_model': cyto,
        # 'spatial_hp_cp': 0,

        ######## Extraction/Neuropil ########
        # 'nbinned': 5000,
        # 'inner_neuropil_radius': 2,
        # 'min_neuropil_pixels': 350,
        # 'lam_percentile': 50.0,

        ######## Classify/Deconv ########
        # 'soma_crop': True,
        # 'spikedetect': True,
        # 'win_baseline': 60.0,
        # 'sig_baseline': 10.0,
        # 'neucoeff': 0.7,

    })
    
    print(f"Running suite2p for channel {functional_chan}...")
    output_ops = run_s2p(ops=ops)
    print(f"Suite2p analysis completed for channel {functional_chan}")
    
    # suite2p_dir = os.path.join(channel_dir, 'suite2p')
    # if os.path.exists(suite2p_dir):
    #     plane0_dir = os.path.join(suite2p_dir, 'plane0')
    #     if os.path.exists(plane0_dir):
    #         for file in os.listdir(plane0_dir):
    #             file_path = os.path.join(plane0_dir, file)
    #             if file.endswith('.bin'):
    #                 os.remove(file_path)
    #             elif file.startswith('temp'):
    #                 os.remove(file_path)

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

    processed_file = convert_tiff_sequence(input_file, output_dir)

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
    
    print("Starting suite2p analysis for channel 1...")
    run_suite2p_channel(output_dir, ch1_dir, 1)
    
    print("Starting suite2p analysis for channel 2...")
    run_suite2p_channel(output_dir, ch2_dir, 2)
    
    print("All analyses completed!")
    print(f"Results saved in:")
    print(f"Channel 1: {ch1_dir}")
    print(f"Channel 2: {ch2_dir}")