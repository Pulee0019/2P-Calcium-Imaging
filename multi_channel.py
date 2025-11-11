# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 10:41:48 2025

@author: Pulee
"""

import numpy as np
import sys
import os
import suite2p
from suite2p.run_s2p import run_s2p
import shutil

# Configuration options - ensure all necessary keys exist
ops = {
    # File paths
    'look_one_level_down': False,
    'fast_disk': [],
    'delete_bin': False,
    'h5py_key': 'data',
    
    # Main settings
    'nplanes': 1,
    'nchannels': 2,
    'functional_chan': 1,  # Channel used for functional ROI extraction
    'diameter': 12,
    'tau': 1.2,
    'fs': 15.136,
    
    # Output settings
    'save_mat': False,
    'combined': False,
    
    # Parallel processing settings
    'num_workers': 0,
    'num_workers_roi': -1,
    
    # Registration settings
    'do_registration': True,
    'nimg_init': 200,
    'batch_size': 200,
    'maxregshift': 0.1,
    'align_by_chan': 1,  # Channel used for registration
    'reg_tif': False,
    'subpixel': 10,
    
    # Cell detection settings
    'connected': True,
    'navg_frames_svd': 5000,
    'nsvd_for_roi': 1000,
    'max_iterations': 20,
    'ratio_neuropil': 6.0,
    'ratio_neuropil_to_cell': 3,
    'tile_factor': 1.0,
    'threshold_scaling': 1.0,
    'max_overlap': 0.75,
    'inner_neuropil_radius': 2,
    'outer_neuropil_radius': np.inf,
    'min_neuropil_pixels': 350,
    
    # Deconvolution settings
    'baseline': 'maximin',
    'win_baseline': 60.0,
    'sig_baseline': 10.0,
    'prctile_baseline': 8.0,
    'neucoeff': 0.7,
    'allow_overlap': False,
    
    # Key multi-channel settings
    'keep_movie_raw': True,       # Keep raw movie data
    'save_path0': '',             # Will be auto-set during runtime
    'reg_file_chan2': '',         # Registration file for channel 2
    'meanImg_chan2': None,        # Mean image for channel 2
    'chan2_thres': 0.5,           # Cell detection threshold for channel 2
    'chan2_cell_detect': True,    # Enable cell detection on channel 2
}

# Database configuration
db = {
    'h5py': [],
    'h5py_key': 'data',
    'look_one_level_down': False,
    'data_path': [r'D:\Expriment\Calcium\test1\output'],
    'subfolders': []
}

### 1. Run full pipeline on first channel
opsEnd = run_s2p(ops=ops, db=db)

# Check and handle return value
if isinstance(opsEnd, dict):
    ops_list = [opsEnd]  # If single plane, convert to list
elif isinstance(opsEnd, list):
    ops_list = opsEnd    # If multi-plane, use list directly
else:
    raise RuntimeError("run_s2p returned unexpected result type")

# Get main save path
if 'save_path0' in ops_list[0]:
    save_path0 = ops_list[0]['save_path0']
else:
    # Derive save_path0 if not present
    save_path0 = os.path.dirname(os.path.dirname(ops_list[0]['save_path']))

### 2. Setup for processing second channel
ops1 = []
j = 0
for ops_plane in ops_list:
    # Create new directory for channel 2
    chan2_path = os.path.join(save_path0, 'chan2')
    plane_path = os.path.join(chan2_path, 'suite2p', f'plane{j}')
    os.makedirs(plane_path, exist_ok=True)
    
    # Update path settings
    ops_plane['save_path0'] = chan2_path
    ops_plane['save_path'] = plane_path
    ops_plane['ops_path'] = os.path.join(plane_path, 'ops.npy')
    
    # Configure for channel 2 processing
    ops_plane['functional_chan'] = 2  # Switch to channel 2
    ops_plane['do_registration'] = False  # Skip registration
    ops_plane['roidetect'] = True  # Perform ROI detection
    
    # Swap channel images
    if 'meanImg_chan2' in ops_plane and ops_plane['meanImg_chan2'] is not None:
        ops_plane['meanImg'] = ops_plane['meanImg_chan2']
    else:
        print(f"Warning: Plane {j} missing meanImg_chan2")
    
    # Set binary file paths
    if 'fast_disk' in ops_plane and ops_plane['fast_disk']:
        fast_disk_path = os.path.join(ops_plane['fast_disk'], 'chan2')
    else:
        fast_disk_path = chan2_path
        
    bin_path = os.path.join(fast_disk_path, 'suite2p', f'plane{j}')
    os.makedirs(bin_path, exist_ok=True)
    ops_plane['fast_disk'] = bin_path
    ops_plane['reg_file'] = os.path.join(bin_path, 'data.bin')
    
    # Copy channel 2 registration file
    if 'reg_file_chan2' in ops_plane and os.path.exists(ops_plane['reg_file_chan2']):
        shutil.copyfile(ops_plane['reg_file_chan2'], ops_plane['reg_file'])
        print(f'Copied registration file to: {ops_plane["reg_file"]}')
    else:
        print(f"Error: Missing channel 2 registration file: {ops_plane.get('reg_file_chan2', '')}")
        continue
    
    # Save configuration
    np.save(ops_plane['ops_path'], ops_plane)
    ops1.append(ops_plane.copy())
    j += 1

# Run processing on second channel if configuration succeeded
if ops1:
    # Save multi-plane configuration
    np.save(os.path.join(ops1[0]['save_path0'], 'suite2p', 'ops1.npy'), ops1)
    
    # Prepare run configuration
    ops_run = ops1[0].copy()
    # Remove plane-specific keys
    for key in ['save_path', 'ops_path', 'reg_file', 'reg_file_chan2']:
        if key in ops_run:
            del ops_run[key]
    
    # Run processing on channel 2
    opsEnd_chan2 = run_s2p(ops=ops_run, db=db)
    
    print("Dual-channel processing completed!")
    print(f"Channel 1 results: {save_path0}")
    print(f"Channel 2 results: {chan2_path}")
else:
    print("Error: Channel 2 configuration failed, processing not run")