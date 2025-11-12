# -*- coding: utf-8 -*-
"""
Created on Tue Nov 4 20:18:50 2025

Modified Calcium Imaging Analyzer with Menu Bar Interface

@author: Pulee
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os
import h5py
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import glob
from matplotlib import colors as mcolors
from scipy.ndimage import generic_filter
import networkx as nx
from scipy import signal
from scipy.stats import zscore
import struct
from logger import log_message, set_log_widget

class CalciumImagingAnalyzer:
    def __init__(self):
        self.base_dir = ""
        self.running_path = ""
        self.sync_folder = ""
        self.experiment_xml_file = ""
        self.OVERLAP_THRESHOLD = 0.5
        self.loaded = False
        self.selected_neuron_idx = -1
        self.neuron_queue = []
        self.background_image = None
        
        self.channel_colors = {
            'ch1': 'green',
            'ch2': 'red',
        }
        
    def auto_detect_files(self, base_directory):
        result = {
            'base_dir': base_directory,
            'channels': {},
            'running_path': None,
            'sync_folder': None,
            'experiment_xml': None,
            'real_time_xml': None
        }
        
        output_dir = os.path.join(base_directory, 'output')
        if not os.path.exists(output_dir):
            for root, dirs, files in os.walk(base_directory):
                if 'output' in dirs:
                    output_dir = os.path.join(root, 'output')
                    break
        
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                if item.startswith('ch') and os.path.isdir(os.path.join(output_dir, item)):
                    channel_path = os.path.join(output_dir, item, 'suite2p', 'plane0')
                    if os.path.exists(channel_path):
                        result['channels'][item] = channel_path
        
        running_files_dat2 = glob.glob(os.path.join(base_directory, '**', '*running*.dat2'), recursive=True)
        running_files_ast2 = glob.glob(os.path.join(base_directory, '**', '*speed*.AST2'), recursive=True)
        
        running_files = running_files_dat2 + running_files_ast2
        
        if running_files:
            if running_files_dat2:
                result['running_path'] = running_files_dat2[0]
                result['running_format'] = 'dat2'
            elif running_files_ast2:
                result['running_path'] = running_files_ast2[0]
                result['running_format'] = 'ast2'
        
        sync_dirs = []
        for root, dirs, files in os.walk(base_directory):
            if 'SyncData' in root:
                sync_dirs.append(root)
        
        if sync_dirs:
            sync_nums = []
            for sync_dir in sync_dirs:
                dir_name = os.path.basename(sync_dir)
                if 'SyncData_' in dir_name:
                    try:
                        num = int(dir_name.split('_')[-1])
                        sync_nums.append((num, sync_dir))
                    except ValueError:
                        continue
            
            if sync_nums:
                sync_nums.sort(key=lambda x: x[0])
                result['sync_folder'] = sync_nums[-1][1]
        
        experiment_xml_files = glob.glob(os.path.join(base_directory, '**', 'Experiment.xml'), recursive=True)
        if experiment_xml_files:
            day_xmls = [f for f in experiment_xml_files if 'day' in os.path.basename(os.path.dirname(f)).lower()]
            if day_xmls:
                result['experiment_xml'] = day_xmls[0]
            else:
                result['experiment_xml'] = experiment_xml_files[0]
        
        if result['sync_folder']:
            real_time_xml = os.path.join(result['sync_folder'], 'ThorRealTimeDataSettings.xml')
            if os.path.exists(real_time_xml):
                result['real_time_xml'] = real_time_xml
        
        return result
    
    def read_ast2_data(self, file_path):
        """Read AST2 data file based on the actual format"""
        try:
            with open(file_path, 'rb') as fileID:
                # Read header according to the actual AST2 format
                # Identifier (4 bytes): "AST2"
                identifier = fileID.read(4)
                if identifier != b"AST2":
                    log_message(f"Not a valid AST2 file. Identifier: {identifier}", "ERROR")
                    return None
                
                # Version (2 bytes): 1.0 (stored as 10)
                version_bytes = fileID.read(2)
                version = struct.unpack("<H", version_bytes)[0] / 10.0  # Convert back to 1.0
                log_message(f"Version: {version}")
                
                # Number of channels (2 bytes)
                num_channels = struct.unpack("<H", fileID.read(2))[0]
                log_message(f"Number of channels: {num_channels}")
                
                # Sample rate (4 bytes, float)
                sample_rate = struct.unpack("<f", fileID.read(4))[0]
                log_message(f"Sample rate: {sample_rate} Hz")
                
                # Downsample factor (4 bytes, int)
                downsample_factor = struct.unpack("<I", fileID.read(4))[0]
                log_message(f"Downsample factor: {downsample_factor}")
                
                # Start time (20 bytes: YYYYMMDDHHMMSSmmm)
                start_time_bytes = fileID.read(20)
                start_time_str = start_time_bytes.decode('utf-8').rstrip('\x00')
                log_message(f"Start time: {start_time_str}")
                
                # Reserved (40 bytes for future expansion)
                reserved = fileID.read(40)
                
                # Now read the data frames
                # Each frame contains:
                # - Timestamp (8 bytes, double)
                # - Speed data for each channel (4 bytes/float per channel)
                frame_size = 8 + 4 * num_channels  # 8 bytes timestamp + 4 bytes per channel
                
                # Calculate number of frames based on remaining file size
                current_pos = fileID.tell()
                fileID.seek(0, 2)  # Seek to end
                file_size = fileID.tell()
                fileID.seek(current_pos)  # Seek back to data start
                
                num_frames = (file_size - current_pos) // frame_size
                log_message(f"Number of frames: {num_frames}")
                
                # Preallocate arrays
                timestamps = np.zeros(num_frames)
                data = np.zeros((num_frames, num_channels))
                
                # Read all frames
                for i in range(num_frames):
                    # Read timestamp (8 bytes, double)
                    timestamp = struct.unpack("<d", fileID.read(8))[0]
                    timestamps[i] = timestamp
                    
                    # Read channel data
                    for channel in range(num_channels):
                        speed = struct.unpack("<f", fileID.read(4))[0]
                        data[i, channel] = speed
                
                return {
                    'version': version,
                    'num_channels': num_channels,
                    'sample_rate': sample_rate,
                    'downsample_factor': downsample_factor,
                    'start_time': start_time_str,
                    'timestamps': timestamps,
                    'data': data
                }
                
        except Exception as e:
            log_message(f"Error reading AST2 file: {str(e)}", "ERROR")
            return None
    
    def compute_mean_image(self, movie):
        mean_img = movie.mean(axis=0).astype(np.float32)
        p5, p95 = np.percentile(mean_img, [5, 95])
        mean_img = (mean_img - p5) / (p95 - p5)
        mean_img = np.clip(mean_img, 0, 1)
        return mean_img

    def load_background_image(self, channel_path):
        try:
            data_bin_path = os.path.join(channel_path, 'data.bin')
            if not os.path.exists(data_bin_path):
                log_message(f"data.bin not found at {data_bin_path}", "WARNING")
                return False
                
            ops_path = os.path.join(channel_path, 'ops.npy')
            if not os.path.exists(ops_path):
                log_message(f"ops.npy not found at {ops_path}", "WARNING")
                return False
                
            ops = np.load(ops_path, allow_pickle=True).item()
            Ly, Lx = ops['Ly'], ops['Lx']
            dtype = np.dtype(ops.get('data_type', 'uint16'))
            
            raw = np.fromfile(data_bin_path, dtype=dtype)
            n_frames = raw.size // (Ly * Lx)
            
            if n_frames == 0:
                log_message("No frames found in data.bin", "WARNING")
                return False
                
            movie = raw.reshape((n_frames, Ly, Lx))

            self.background_image = self.compute_mean_image(movie)

            log_message(f"Background image loaded: {self.background_image.shape}")
            return True
            
        except Exception as e:
            log_message(f"Error loading background image: {str(e)}", "ERROR")
            return False
        
    def load_from_directory(self, base_directory):
        try:
            file_paths = self.auto_detect_files(base_directory)
            
            missing_files = []
            if not file_paths['channels']:
                missing_files.append('channel data')
            if not file_paths['running_path']:
                missing_files.append('running data')
            if not file_paths['sync_folder']:
                missing_files.append('sync folder')
            if not file_paths['experiment_xml']:
                missing_files.append('experiment XML')
            
            if missing_files:
                return False, f"Missing required files: {', '.join(missing_files)}"
            
            self.base_dir = file_paths['base_dir']
            self.running_path = file_paths['running_path']
            self.running_format = file_paths.get('running_format', 'dat2')
            self.sync_folder = file_paths['sync_folder']
            self.experiment_xml_file = file_paths['experiment_xml']
            
            self.channels_data = {}
            channel_names = sorted(file_paths['channels'].keys())
            
            for channel_name in channel_names:
                channel_path = file_paths['channels'][channel_name]
                
                self.channels_data[channel_name] = {
                    'stat': np.load(os.path.join(channel_path, "stat.npy"), allow_pickle=True),
                    'ops': np.load(os.path.join(channel_path, "ops.npy"), allow_pickle=True).item(),
                    'F': np.load(os.path.join(channel_path, "F.npy")),
                    'Fneu': np.load(os.path.join(channel_path, "Fneu.npy")),
                    'spks': np.load(os.path.join(channel_path, "spks.npy")),
                    'iscell': np.load(os.path.join(channel_path, "iscell.npy"))
                }
                if channel_name == channel_names[0]:
                    self.load_background_image(channel_path)
            
            first_channel = list(self.channels_data.values())[0]
            self.Ly = first_channel['ops']['Ly']
            self.Lx = first_channel['ops']['Lx']
            
            for channel_name, data in self.channels_data.items():
                if data['ops']['Ly'] != self.Ly or data['ops']['Lx'] != self.Lx:
                    return False, f"Channel {channel_name} dimensions don't match!"
            
            if self.running_format == 'dat2':
                data = np.fromfile(self.running_path, dtype=np.float64)
                data = data.reshape(-1, 100, 5)
                self.running_data = -data[:, :, 1].ravel() * 7.5 / 360 * np.pi
                raw_running_time = data[:, :, 0].ravel()
                self.running_relative_time = (raw_running_time - raw_running_time[0]) * 24 * 60
            elif self.running_format == 'ast2':
                ast2_data = self.read_ast2_data(self.running_path)
                if ast2_data is None:
                    return False, "Failed to read AST2 running data"
                
                self.running_data = ast2_data['data'][:, 0]
                raw_running_time = ast2_data['timestamps']
                self.running_relative_time = (raw_running_time - raw_running_time[0]) / 60
                
                log_message(f"AST2 data loaded: {len(self.running_data)} samples, sample rate: {ast2_data['sample_rate']} Hz")
            else:
                return False, f"Unsupported running data format: {self.running_format}"
            
            sync_file = os.path.join(self.sync_folder, "Episode_0000.h5")
            with h5py.File(sync_file, 'r') as h5f:
                self.running_sync = np.array(h5f['/AI/Runningdata'])
                self.frame_out = np.array(h5f['/DI/FrameOut'])
                self.trigger_signal = np.array(h5f['/DI/Triggersignal'])
            
            real_time_xml_file = os.path.join(self.sync_folder, "ThorRealTimeDataSettings.xml")
            tree = ET.parse(real_time_xml_file)
            root = tree.getroot()
            
            daq = root.find(".//AcquireBoard[@active='1']")
            if daq is None:
                raise RuntimeError("No active DAQ board found in ThorRealTimeDataSettings.xml")  

            sr_node = daq.find(".//SampleRate[@enable='1']")
            if sr_node is None:
                raise RuntimeError("No enabled <SampleRate> found in active DAQ board")

            self.sample_rate = float(sr_node.get("rate"))

            tree = ET.parse(self.experiment_xml_file)
            root = tree.getroot()    
            lsm = root.find('.//LSM[@name="ResonanceGalvo"]')
            
            if lsm is not None:
                self.frame_rate = float(lsm.get('frameRate'))
                self.average_num = int(lsm.get('averageNum'))
            else:
                log_message("LSM cannot find!", "WARNING")
                self.frame_rate = 10.0
                self.average_num = 1
            
            running_onset_idx = np.where(self.trigger_signal > 0.5)[0]
            imaging_onset_idx = np.where(self.frame_out > 0.5)[0]
            
            if len(running_onset_idx) > 0 and len(imaging_onset_idx) > 0:
                running_onset = running_onset_idx[0]
                diff = np.diff(imaging_onset_idx)
                breaks = np.where(diff > 100)[-1]
                starts = np.concatenate(([0], breaks + 1))
                first_indices = imaging_onset_idx[starts]
                imaging_onset = first_indices[-1]
                self.relative_time = (imaging_onset - running_onset) / self.sample_rate
            else:
                self.relative_time = 0.0
                log_message("Trigger signals not found, using zero offset", "WARNING")
            
            self.running_time = self.running_relative_time * 60 - self.relative_time
            
            plt.figure()
            plt.plot(self.trigger_signal, color="#0004FF", label='trigger signal')
            plt.plot(self.frame_out, color="#3CFF00", label='frame out signal')
            plt.axvline(running_onset, color='#000000', label='running onset')
            plt.axvline(imaging_onset, color="#FF0000", label='imaging onset')
            plt.legend(loc='upper left')
            plt.show()
            
            self.neurons_by_channel = {}
            for channel_name, data in self.channels_data.items():
                self.neurons_by_channel[channel_name] = self.preprocess_neurons(
                    data['stat'], data['iscell'], channel_name
                )
            
            if len(self.channels_data) > 1:
                self.match_neurons()
            else:
                self.matched_neurons = {channel_name: set() for channel_name in self.channels_data.keys()}
                self.overlap_pairs = []
            
            self.create_combined_mask()
            
            self.combined_activity = self.combine_activity_data()
            
            first_channel_name = list(self.channels_data.keys())[0]
            n_frames = self.channels_data[first_channel_name]['F'].shape[1]
            self.imaging_times = np.arange(n_frames) / self.frame_rate * self.average_num
            
            valid_mask = (self.running_time >= 0) & (self.running_time <= self.imaging_times[-1])
            self.running_time_valid = self.running_time[valid_mask]
            self.running_data_valid = self.running_data[valid_mask]
            
            # Calculate acceleration and classify movement types
            self.calculate_acceleration()
            self.classify_movement_types()
            
            self.loaded = True
            log_message("Data loaded successfully!")
            log_message(f"Total channels: {len(self.channels_data)}")
            log_message(f"Total neurons: {len(self.combined_neurons)}")
            
            for channel_name in self.channels_data.keys():
                channel_only = len([n for n in self.combined_neurons if n['type'] == f'{channel_name}_only'])
                log_message(f"{channel_name} only: {channel_only}")
            
            if len(self.channels_data) > 1:
                log_message(f"Overlap: {len(self.overlap_pairs)}")
            
            log_message(f"Movement periods: {len(self.movement_periods)}")
            log_message(f"Rest periods: {len(self.rest_periods)}")
            log_message(f"Movement onsets: {len(self.movement_onsets)}")
            log_message(f"Jerks: {len(self.jerks)}")
            log_message(f"Locomotion initiations: {len(self.locomotion_initiations)}")
            log_message(f"Continuous locomotion periods: {len(self.continuous_locomotion_periods)}")
            log_message(f"Locomotion terminations: {len(self.locomotion_terminations)}")
            
            self.neuron_queue = [0]
            self.selected_neuron_idx = 0
            
            return True, "Data loaded successfully!"
            
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def calculate_acceleration(self):
        """Calculate acceleration from velocity data"""
        if len(self.running_time_valid) > 1:
            dt = np.diff(self.running_time_valid)
            # Use central difference for acceleration calculation
            self.acceleration = np.zeros_like(self.running_data_valid)
            self.acceleration[1:-1] = (self.running_data_valid[2:] - self.running_data_valid[:-2]) / (dt[1:] + dt[:-1])
            # Handle edges
            self.acceleration[0] = (self.running_data_valid[1] - self.running_data_valid[0]) / dt[0]
            self.acceleration[-1] = (self.running_data_valid[-1] - self.running_data_valid[-2]) / dt[-1]
        else:
            self.acceleration = np.zeros_like(self.running_data_valid)
    
    def classify_movement_types(self):
        """Classify movement types based on the provided criteria"""
        # Define thresholds
        MOVEMENT_VELOCITY_THRESHOLD = 1.0  # cm/s
        MOVEMENT_ACCELERATION_THRESHOLD = 40  # cm/s²
        REST_VELOCITY_THRESHOLD = 0.5  # cm/s
        MOVEMENT_ONSET_THRESHOLD = 0.5  # cm/s
        MOVEMENT_ONSET_PEAK_THRESHOLD = 1.0  # cm/s
        JERK_MAX_VELOCITY_THRESHOLD = 2.2  # cm/s
        LOCOMOTION_INITIATION_THRESHOLD = 4.5  # cm/s
        CONTINUOUS_LOCOMOTION_THRESHOLD = 4.5  # cm/s
        LOCOMOTION_TERMINATION_THRESHOLD = 2.2  # cm/s
        
        # Calculate sample rate for running data
        if len(self.running_time_valid) > 1:
            sample_interval = np.mean(np.diff(self.running_time_valid))
            sample_rate = 1.0 / sample_interval
        else:
            sample_rate = 20.0  # default assumption
        
        # Convert time durations to sample counts
        window_1s = int(1.0 * sample_rate)
        window_0_5s = int(0.5 * sample_rate)
        window_2s = int(2.0 * sample_rate)
        window_3s = int(3.0 * sample_rate)
        
        # 1. Movement periods: velocity > 2.2 cm/s and acceleration > 40 cm/s² in ±1s window
        movement_periods_mask = np.zeros_like(self.running_data_valid, dtype=bool)
        for i in range(len(self.running_data_valid)):
            start_idx = max(0, i - window_1s)
            end_idx = min(len(self.running_data_valid), i + window_1s + 1)
            
            # Check if velocity and acceleration exceed thresholds in the window
            vel_window = self.running_data_valid[start_idx:end_idx]
            # acc_window = self.acceleration[start_idx:end_idx]
            
            # if np.all(vel_window > MOVEMENT_VELOCITY_THRESHOLD) and np.all(acc_window > MOVEMENT_ACCELERATION_THRESHOLD):
            #     movement_periods_mask[i] = True
            if np.all(vel_window > MOVEMENT_VELOCITY_THRESHOLD):
                movement_periods_mask[i] = True
        
        self.movement_periods = self.find_contiguous_regions_time(movement_periods_mask, 1)
        
        # 2. Rest periods: no velocity > 0.2 cm/s in ±1s window
        rest_periods_mask = np.zeros_like(self.running_data_valid, dtype=bool)
        for i in range(len(self.running_data_valid)):
            start_idx = max(0, i - window_1s)
            end_idx = min(len(self.running_data_valid), i + window_1s + 1)
            
            # Check if all velocities in the window are below threshold
            vel_window = self.running_data_valid[start_idx:end_idx]
            
            if np.all(np.abs(vel_window) < REST_VELOCITY_THRESHOLD):
                rest_periods_mask[i] = True
        
        self.rest_periods = self.find_contiguous_regions_time(rest_periods_mask, window_1s)
        
        # 3. Movement onsets: positive-going threshold crossings of velocity at 0.2 cm/s
        # where peak velocity reaches at least 1 cm/s within 1s post-onset
        # and no velocities > 0.2 cm/s within 0.5s prior to onset
        self.movement_onsets = []
        velocity_abs = np.abs(self.running_data_valid)
        
        for i in range(1, len(velocity_abs)):
            # Check for positive-going threshold crossing
            if (velocity_abs[i] > MOVEMENT_ONSET_THRESHOLD and 
                velocity_abs[i-1] <= MOVEMENT_ONSET_THRESHOLD):
                
                # Check no velocities > 0.2 cm/s within 0.5s prior
                prior_start = max(0, i - window_0_5s)
                if np.any(velocity_abs[prior_start:i] > MOVEMENT_ONSET_THRESHOLD):
                    continue
                
                # Check peak velocity reaches at least 1 cm/s within 1s post-onset
                post_end = min(len(velocity_abs), i + window_1s)
                if np.max(velocity_abs[i:post_end]) >= MOVEMENT_ONSET_PEAK_THRESHOLD:
                    self.movement_onsets.append(self.running_time_valid[i])
        
        # 4. Jerks: movement onsets where max velocity in 1-2s window is less than 2.2 cm/s
        self.jerks = []
        for onset_time in self.movement_onsets:
            onset_idx = np.argmin(np.abs(self.running_time_valid - onset_time))
            
            # Check window 1-2s after onset
            start_idx = min(len(velocity_abs), onset_idx + window_1s)
            end_idx = min(len(velocity_abs), onset_idx + window_2s)
            
            if start_idx < end_idx and np.max(velocity_abs[start_idx:end_idx]) < JERK_MAX_VELOCITY_THRESHOLD:
                self.jerks.append(onset_time)
        
        # 5. Locomotion initiations: movement onsets where mean velocity in 0.5-2s window > 4.5 cm/s
        self.locomotion_initiations = []
        for onset_time in self.movement_onsets:
            onset_idx = np.argmin(np.abs(self.running_time_valid - onset_time))
            
            # Check window 0.5-2s after onset
            start_idx = min(len(velocity_abs), onset_idx + window_0_5s)
            end_idx = min(len(velocity_abs), onset_idx + window_2s)
            
            if start_idx < end_idx and np.mean(velocity_abs[start_idx:end_idx]) > LOCOMOTION_INITIATION_THRESHOLD:
                self.locomotion_initiations.append(onset_time)
        
        # 6. Continuous locomotion periods: bouts > 3s with velocity sustained above 4.5 cm/s
        continuous_locomotion_mask = velocity_abs > CONTINUOUS_LOCOMOTION_THRESHOLD
        self.continuous_locomotion_periods = self.find_contiguous_regions_time(continuous_locomotion_mask, window_3s)
        
        # 7. Locomotion terminations: cross from continuous locomotion below 2.2 cm/s
        # and remain below 2.2 cm/s for 1s
        self.locomotion_terminations = []
        for start_idx, end_idx in self.continuous_locomotion_periods:
            # Check if velocity drops below 2.2 cm/s at the end
            if end_idx < len(velocity_abs) and velocity_abs[end_idx] < LOCOMOTION_TERMINATION_THRESHOLD:
                # Check if it remains below 2.2 cm/s for 1s
                post_end = min(len(velocity_abs), end_idx + window_1s)
                if np.all(velocity_abs[end_idx:post_end] < LOCOMOTION_TERMINATION_THRESHOLD):
                    self.locomotion_terminations.append(self.running_time_valid[end_idx])
    
    def find_contiguous_regions_time(self, mask, min_samples):
        """Find contiguous regions in a boolean mask with minimum length and return time values"""
        regions = []
        in_region = False
        region_start = 0
        
        for i in range(len(mask)):
            if mask[i] and not in_region:
                in_region = True
                region_start = i
            elif not mask[i] and in_region:
                in_region = False
                region_end = i
                if region_end - region_start >= min_samples:
                    regions.append((region_start, region_end))
        
        # Check if we're still in a region at the end
        if in_region and len(mask) - region_start >= min_samples:
            regions.append((region_start, len(mask)))
        
        # Convert to time values
        time_regions = []
        for start, end in regions:
            start_idx = min(start, len(self.running_time_valid) - 1)
            end_idx = min(end, len(self.running_time_valid) - 1)
            
            if start_idx < len(self.running_time_valid) and end_idx < len(self.running_time_valid):
                time_regions.append((self.running_time_valid[start_idx], self.running_time_valid[end_idx]))
        
        return time_regions
    
    def preprocess_neurons(self, stat, iscell, channel_name):
        neurons = []
        for i, cell in enumerate(stat):
            if iscell[i, 1] > 0.0:
                mask = np.zeros((self.Ly, self.Lx), dtype=bool)
                ypix = cell['ypix'][~cell.get('overlap', np.zeros_like(cell['ypix'], dtype=bool))]
                xpix = cell['xpix'][~cell.get('overlap', np.zeros_like(cell['xpix'], dtype=bool))]
                mask[ypix, xpix] = True
                neurons.append({
                    'mask': mask,
                    'area': len(ypix),
                    'coords': (ypix, xpix),
                    'index': i,
                    'channel': channel_name
                })
        return neurons
    
    def match_neurons(self):
        channel_names = list(self.channels_data.keys())
        self.matched_neurons = {channel: set() for channel in channel_names}
        self.overlap_pairs = []
        
        for i in range(len(channel_names)):
            for j in range(i + 1, len(channel_names)):
                ch1 = channel_names[i]
                ch2 = channel_names[j]
                
                for idx1, n1 in enumerate(self.neurons_by_channel[ch1]):
                    for idx2, n2 in enumerate(self.neurons_by_channel[ch2]):
                        overlap = np.sum(n1['mask'] & n2['mask'])
                        if overlap > 0:
                            min_area = min(n1['area'], n2['area'])
                            overlap_ratio = overlap / min_area
                            if overlap_ratio >= self.OVERLAP_THRESHOLD:
                                self.overlap_pairs.append((ch1, idx1, ch2, idx2, overlap_ratio))
                                self.matched_neurons[ch1].add(idx1)
                                self.matched_neurons[ch2].add(idx2)
    
    def create_combined_mask(self):
        self.combined_neurons = []
        
        if len(self.channels_data) == 1:
            channel_name = list(self.channels_data.keys())[0]
            for i, n in enumerate(self.neurons_by_channel[channel_name]):
                neuron_data = {
                    'mask': n['mask'],
                    'coords': n['coords'],
                    'type': f'{channel_name}_only',
                    'color': self.channel_colors.get(channel_name, 'gray'),
                    'orig_index': n['index'],
                    'orig_channel': channel_name
                }
                self.combined_neurons.append(neuron_data)
        else:
            for channel_name, neurons in self.neurons_by_channel.items():
                for i, n in enumerate(neurons):
                    if i not in self.matched_neurons[channel_name]:
                        neuron_data = {
                            'mask': n['mask'],
                            'coords': n['coords'],
                            'type': f'{channel_name}_only',
                            'color': self.channel_colors.get(channel_name, 'gray'),
                            'orig_index': n['index'],
                            'orig_channel': channel_name
                        }
                        self.combined_neurons.append(neuron_data)
            
            for ch1, idx1, ch2, idx2, ratio in self.overlap_pairs:
                n1 = self.neurons_by_channel[ch1][idx1]
                n2 = self.neurons_by_channel[ch2][idx2]
                
                if n1['area'] >= n2['area']:
                    selected = n1
                    channel = ch1
                    orig_index = n1['index']
                else:
                    selected = n2
                    channel = ch2
                    orig_index = n2['index']
                
                neuron_data = {
                    'mask': selected['mask'],
                    'coords': selected['coords'],
                    'type': 'overlap',
                    'color': 'yellow',
                    'orig_index': orig_index,
                    'orig_channel': channel,
                    'overlap_ratio': ratio,
                    'orig_channels': {ch1: n1['index'], ch2: n2['index']}
                }
                self.combined_neurons.append(neuron_data)
        
        self.combined_mask = np.zeros((self.Ly, self.Lx), dtype=int)
        for idx, neuron in enumerate(self.combined_neurons):
            ypix, xpix = neuron['coords']
            self.combined_mask[ypix, xpix] = idx + 1
        
        colors = ['black']
        for neuron in self.combined_neurons:
            colors.append(neuron['color'])
        self.cmap = ListedColormap(colors)
    
    def combine_activity_data(self):
        combined_activity = []
        
        for neuron in self.combined_neurons:
            neuron_data = {
                'type': neuron['type'],
                'mask': neuron['mask'],
                'coords': neuron['coords'],
                'color': neuron.get('color', 'gray'),
                'index': len(combined_activity)
            }
            
            if neuron['type'].endswith('_only'):
                channel_name = neuron['orig_channel']
                idx = neuron['orig_index']
                
                neuron_data['F'] = self.channels_data[channel_name]['F'][idx]
                neuron_data['Fneu'] = self.channels_data[channel_name]['Fneu'][idx]
                neuron_data['spks'] = self.channels_data[channel_name]['spks'][idx]
                neuron_data['channel'] = channel_name
                
            elif neuron['type'] == 'overlap':
                for channel_name, idx in neuron['orig_channels'].items():
                    neuron_data[f'F_{channel_name}'] = self.channels_data[channel_name]['F'][idx]
                    neuron_data[f'Fneu_{channel_name}'] = self.channels_data[channel_name]['Fneu'][idx]
                    neuron_data[f'spks_{channel_name}'] = self.channels_data[channel_name]['spks'][idx]
                
                neuron_data['overlap_ratio'] = neuron['overlap_ratio']
                neuron_data['channels'] = list(neuron['orig_channels'].keys())
            
            combined_activity.append(neuron_data)
        
        return combined_activity
    
    def calculate_dff(self, F, Fneu, window_size=600, percentile_value=20):
        """Calculate dF/F from fluorescence data"""
        F0 = generic_filter(Fneu, function=lambda x: np.percentile(x, percentile_value), size=window_size)
        dff = (Fneu - F0) / F0
        return dff
    
    def plot_neuron_activity(self, neuron_idx, data_type='dff', movement_types=None):
        if not self.loaded or neuron_idx < 0 or neuron_idx >= len(self.combined_activity):
            return None
        
        if movement_types is None:
            movement_types = []
        
        neuron = self.combined_activity[neuron_idx]
        neuron_type = neuron['type']
        
        fig = Figure(figsize=(8, 3), dpi=100)
        ax = fig.add_subplot(111)

        if neuron_type.endswith('_only'):
            channel_name = neuron['channel']
            color = self.channel_colors.get(channel_name, 'gray')
            
            if data_type == 'F':
                data = neuron['F']
                ylabel = 'Fluorescence (F)'
            elif data_type == 'Fneu':
                data = neuron['Fneu']
                ylabel = 'Fluorescence neuropil (Fneu)'
            elif data_type == 'dff':
                data = self.calculate_dff(neuron['F'], neuron['Fneu'])
                ylabel = 'dF/F'
            elif data_type == 'spks':
                data = neuron['spks']
                ylabel = 'Spikes'
            else:
                data = self.calculate_dff(neuron['F'], neuron['Fneu'])
                ylabel = 'dF/F'
            
            ax.plot(self.imaging_times, data, color=color, 
                   label=f'Neuron {neuron_idx} ({channel_name})', alpha=0.8)
            ax.set_ylabel(ylabel, color=color)
            
        else:  # overlap
            for channel_name in neuron['channels']:
                color = self.channel_colors.get(channel_name, 'gray')
                
                if data_type == 'F':
                    data = neuron[f'F_{channel_name}']
                    ylabel = 'Fluorescence (F)'
                elif data_type == 'Fneu':
                    data = neuron[f'Fneu_{channel_name}']
                    ylabel = 'Fluorescence neuropil (Fneu)'
                elif data_type == 'dff':
                    data = self.calculate_dff(neuron[f'F_{channel_name}'], neuron[f'Fneu_{channel_name}'])
                    ylabel = 'dF/F'
                elif data_type == 'spks':
                    data = neuron[f'spks_{channel_name}']
                    ylabel = 'Spikes'
                else:
                    data = self.calculate_dff(neuron[f'F_{channel_name}'], neuron[f'Fneu_{channel_name}'])
                    ylabel = 'dF/F'
                
                ax.plot(self.imaging_times, data, color=color, 
                       label=f'Neuron {neuron_idx} ({channel_name})', alpha=0.7)
            
            ax.set_ylabel(ylabel)
        
        # Add selected movement event markers
        if 'movement_periods' in movement_types and hasattr(self, 'movement_periods'):
            for bout_start, bout_end in self.movement_periods:
                ax.axvspan(bout_start, bout_end, alpha=0.2, color='blue', label='Movement Period' if 'Movement Period' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'rest_periods' in movement_types and hasattr(self, 'rest_periods'):
            for rest_start, rest_end in self.rest_periods:
                ax.axvspan(rest_start, rest_end, alpha=0.2, color='gray', label='Rest Period' if 'Rest Period' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'movement_onsets' in movement_types and hasattr(self, 'movement_onsets'):
            for onset_time in self.movement_onsets:
                ax.axvline(x=onset_time, color='orange', linestyle='--', alpha=0.7, label='Movement Onset' if 'Movement Onset' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'jerks' in movement_types and hasattr(self, 'jerks'):
            for jerk_time in self.jerks:
                ax.axvline(x=jerk_time, color='purple', linestyle=':', alpha=0.7, label='Jerk' if 'Jerk' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'locomotion_initiations' in movement_types and hasattr(self, 'locomotion_initiations'):
            for initiation_time in self.locomotion_initiations:
                ax.axvline(x=initiation_time, color='cyan', linestyle='-.', alpha=0.7, label='Locomotion Initiation' if 'Locomotion Initiation' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'continuous_locomotion_periods' in movement_types and hasattr(self, 'continuous_locomotion_periods'):
            for loco_start, loco_end in self.continuous_locomotion_periods:
                ax.axvspan(loco_start, loco_end, alpha=0.2, color='green', label='Continuous Locomotion' if 'Continuous Locomotion' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'locomotion_terminations' in movement_types and hasattr(self, 'locomotion_terminations'):
            for termination_time in self.locomotion_terminations:
                ax.axvline(x=termination_time, color='brown', linestyle='--', alpha=0.7, label='Locomotion Termination' if 'Locomotion Termination' not in [l.get_label() for l in ax.get_lines()] else "")
        
        ax.set_title(f'Neuron {neuron_idx} Activity ({neuron_type}) - {data_type}', fontsize=14)
        ax.grid(False)
        ax.legend(loc='upper right')
        
        min_time = min(self.imaging_times[0], self.running_time_valid[0])
        max_time = max(self.imaging_times[-1], self.running_time_valid[-1])
        ax.set_xlim(min_time, max_time)
        
        fig.tight_layout()
        return fig
    
    def plot_multiple_neurons_activity(self, neuron_indices, data_type='dff', movement_types=None):
        if not self.loaded or not neuron_indices:
            return None
        
        if movement_types is None:
            movement_types = []
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        fig = Figure(figsize=(8, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        for i, neuron_idx in enumerate(neuron_indices):
            if neuron_idx < 0 or neuron_idx >= len(self.combined_activity):
                continue
                
            neuron = self.combined_activity[neuron_idx]
            color = colors[i % len(colors)]
            
            if neuron['type'].endswith('_only'):
                channel_name = neuron['channel']
                
                if data_type == 'F':
                    data = neuron['F']
                elif data_type == 'Fneu':
                    data = neuron['Fneu']
                elif data_type == 'dff':
                    data = self.calculate_dff(neuron['F'], neuron['Fneu'])
                elif data_type == 'spks':
                    data = neuron['spks']
                else:
                    data = self.calculate_dff(neuron['F'], neuron['Fneu'])
                
                ax.plot(self.imaging_times, data, color=color, 
                       label=f'Neuron {neuron_idx} ({channel_name})', alpha=0.8)
                
            else:  # overlap
                first_channel = neuron['channels'][0]
                
                if data_type == 'F':
                    data = neuron[f'F_{first_channel}']
                elif data_type == 'Fneu':
                    data = neuron[f'Fneu_{first_channel}']
                elif data_type == 'dff':
                    data = self.calculate_dff(neuron[f'F_{first_channel}'], neuron[f'Fneu_{first_channel}'])
                elif data_type == 'spks':
                    data = neuron[f'spks_{first_channel}']
                else:
                    data = self.calculate_dff(neuron[f'F_{first_channel}'], neuron[f'Fneu_{first_channel}'])
                
                # ax.plot(self.imaging_times, data, color=color, 
                #        label=f'Neuron {neuron_idx} (overlap)', alpha=0.8)
                ax.plot(self.imaging_times, data, color=color, alpha=0.8)
        
        # Add selected movement event markers
        if 'movement_periods' in movement_types and hasattr(self, 'movement_periods'):
            for bout_start, bout_end in self.movement_periods:
                ax.axvspan(bout_start, bout_end, alpha=0.2, color='blue', label='Movement Period' if 'Movement Period' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'rest_periods' in movement_types and hasattr(self, 'rest_periods'):
            for rest_start, rest_end in self.rest_periods:
                ax.axvspan(rest_start, rest_end, alpha=0.2, color='gray', label='Rest Period' if 'Rest Period' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'movement_onsets' in movement_types and hasattr(self, 'movement_onsets'):
            for onset_time in self.movement_onsets:
                ax.axvline(x=onset_time, color='orange', linestyle='--', alpha=0.7, label='Movement Onset' if 'Movement Onset' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'jerks' in movement_types and hasattr(self, 'jerks'):
            for jerk_time in self.jerks:
                ax.axvline(x=jerk_time, color='purple', linestyle=':', alpha=0.7, label='Jerk' if 'Jerk' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'locomotion_initiations' in movement_types and hasattr(self, 'locomotion_initiations'):
            for initiation_time in self.locomotion_initiations:
                ax.axvline(x=initiation_time, color='cyan', linestyle='-.', alpha=0.7, label='Locomotion Initiation' if 'Locomotion Initiation' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'continuous_locomotion_periods' in movement_types and hasattr(self, 'continuous_locomotion_periods'):
            for loco_start, loco_end in self.continuous_locomotion_periods:
                ax.axvspan(loco_start, loco_end, alpha=0.2, color='green', label='Continuous Locomotion' if 'Continuous Locomotion' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'locomotion_terminations' in movement_types and hasattr(self, 'locomotion_terminations'):
            for termination_time in self.locomotion_terminations:
                ax.axvline(x=termination_time, color='brown', linestyle='--', alpha=0.7, label='Locomotion Termination' if 'Locomotion Termination' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if data_type == 'F':
            ylabel = 'Fluorescence (F)'
        elif data_type == 'Fneu':
            ylabel = 'Fluorescence neuropil (Fneu)'
        elif data_type == 'dff':
            ylabel = 'dF/F'
        elif data_type == 'spks':
            ylabel = 'Spikes'
        else:
            ylabel = 'dF/F'
            
        ax.set_ylabel(ylabel)
        ax.set_title(f'Multiple Neurons Activity - {data_type}', fontsize=14)
        ax.grid(False)
        # ax.legend(loc='upper right')
        
        min_time = min(self.imaging_times[0], self.running_time_valid[0])
        max_time = max(self.imaging_times[-1], self.running_time_valid[-1])
        ax.set_xlim(min_time, max_time)
        
        fig.tight_layout()
        return fig
    
    def plot_running_activity(self, movement_types=None):
        if not self.loaded:
            return None
        
        if movement_types is None:
            movement_types = []
        
        fig = Figure(figsize=(8, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot running speed
        ax.plot(self.running_time_valid, self.running_data_valid, 'b-', alpha=0.7, label='Running Speed')
        ax.set_ylabel('Running Speed(cm/s)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # # Plot acceleration
        # ax2 = ax.twinx()
        # ax2.plot(self.running_time_valid, self.acceleration, 'r-', alpha=0.7, label='Acceleration')
        # ax2.set_ylabel('Acceleration(cm/s²)', color='r')
        # ax2.tick_params(axis='y', labelcolor='r')
        
        # Add selected movement event markers
        if 'movement_periods' in movement_types and hasattr(self, 'movement_periods'):
            for bout_start, bout_end in self.movement_periods:
                ax.axvspan(bout_start, bout_end, alpha=0.2, color='blue', label='Movement Period' if 'Movement Period' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'rest_periods' in movement_types and hasattr(self, 'rest_periods'):
            for rest_start, rest_end in self.rest_periods:
                ax.axvspan(rest_start, rest_end, alpha=0.2, color='gray', label='Rest Period' if 'Rest Period' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'movement_onsets' in movement_types and hasattr(self, 'movement_onsets'):
            for onset_time in self.movement_onsets:
                ax.axvline(x=onset_time, color='orange', linestyle='--', alpha=0.7, label='Movement Onset' if 'Movement Onset' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'jerks' in movement_types and hasattr(self, 'jerks'):
            for jerk_time in self.jerks:
                ax.axvline(x=jerk_time, color='purple', linestyle=':', alpha=0.7, label='Jerk' if 'Jerk' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'locomotion_initiations' in movement_types and hasattr(self, 'locomotion_initiations'):
            for initiation_time in self.locomotion_initiations:
                ax.axvline(x=initiation_time, color='cyan', linestyle='-.', alpha=0.7, label='Locomotion Initiation' if 'Locomotion Initiation' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'continuous_locomotion_periods' in movement_types and hasattr(self, 'continuous_locomotion_periods'):
            for loco_start, loco_end in self.continuous_locomotion_periods:
                ax.axvspan(loco_start, loco_end, alpha=0.2, color='green', label='Continuous Locomotion' if 'Continuous Locomotion' not in [l.get_label() for l in ax.get_lines()] else "")
        
        if 'locomotion_terminations' in movement_types and hasattr(self, 'locomotion_terminations'):
            for termination_time in self.locomotion_terminations:
                ax.axvline(x=termination_time, color='brown', linestyle='--', alpha=0.7, label='Locomotion Termination' if 'Locomotion Termination' not in [l.get_label() for l in ax.get_lines()] else "")
        
        min_time = min(self.imaging_times[0], self.running_time_valid[0])
        max_time = max(self.imaging_times[-1], self.running_time_valid[-1])
        ax.set_xlim(min_time, max_time)
        ax.set_xlabel('Time (s)')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        fig.tight_layout()
        return fig
    
    def plot_neuron_mask(self, highlight_idx=-1, queue_indices=[]):
        if not self.loaded:
            return None
        
        fig = Figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111)
        
        if self.background_image is not None:
            ax.imshow(self.background_image, cmap='gray', extent=[0, self.Lx, self.Ly, 0])
            
            img = ax.imshow(self.combined_mask, cmap=self.cmap, 
                        vmin=0, vmax=len(self.combined_neurons), 
                        interpolation='nearest', alpha=0.5)
        else:
            img = ax.imshow(self.combined_mask, cmap=self.cmap, 
                        vmin=0, vmax=len(self.combined_neurons), 
                        interpolation='nearest')
        
        if 0 <= highlight_idx < len(self.combined_neurons):
            neuron = self.combined_neurons[highlight_idx]
            ypix, xpix = neuron['coords']
            highlight_mask = np.zeros_like(self.combined_mask)
            highlight_mask[ypix, xpix] = 1
            ax.imshow(highlight_mask, cmap=ListedColormap(['none', 'cyan']), 
                    alpha=0.5, interpolation='nearest')
        
        for neuron_idx in queue_indices:
            if 0 <= neuron_idx < len(self.combined_neurons) and neuron_idx != highlight_idx:
                neuron = self.combined_neurons[neuron_idx]
                ypix, xpix = neuron['coords']
                queue_mask = np.zeros_like(self.combined_mask)
                queue_mask[ypix, xpix] = 1
                ax.imshow(queue_mask, cmap=ListedColormap(['none', 'magenta']), 
                         alpha=0.5, interpolation='nearest')
        
        legend_patches = []
        
        for channel_name in self.channels_data.keys():
            color = self.channel_colors.get(channel_name, 'gray')
            count = len([n for n in self.combined_neurons if n['type'] == f'{channel_name}_only'])
            legend_patches.append(mpatches.Patch(color=color, label=f'{channel_name} Only ({count})'))
        
        if len(self.channels_data) > 1:
            overlap_count = len([n for n in self.combined_neurons if n['type'] == 'overlap'])
            legend_patches.append(mpatches.Patch(color='yellow', label=f'Overlap ({overlap_count})'))
        
        fig.tight_layout()
        ax.set_axis_off() 
        return fig

    def compute_cross_correlation_with_lags(self, data_type='spks', reference_channel='ch1'):
        if reference_channel not in self.channels_data:
            return None, None, None
            
        reference_indices = []
        for i, neuron in enumerate(self.combined_activity):
            if neuron['type'] == f'{reference_channel}_only' or \
               (neuron['type'] == 'overlap' and reference_channel in neuron.get('channels', [])):
                
                if data_type == 'F' and (neuron.get('F') is not None or neuron.get(f'F_{reference_channel}') is not None):
                    reference_indices.append(i)
                elif data_type == 'Fneu' and (neuron.get('Fneu') is not None or neuron.get(f'Fneu_{reference_channel}') is not None):
                    reference_indices.append(i)
                elif data_type == 'dff':
                    if (neuron.get('F') is not None and neuron.get('Fneu') is not None) or \
                       (neuron.get(f'F_{reference_channel}') is not None and neuron.get(f'Fneu_{reference_channel}') is not None):
                        reference_indices.append(i)
                elif data_type == 'spks' and (neuron.get('spks') is not None or neuron.get(f'spks_{reference_channel}') is not None):
                    reference_indices.append(i)
        
        if not reference_indices:
            return None, None, None
        
        n_neurons = len(reference_indices)
        max_lag = 10
        
        cross_corr_matrix = np.zeros((n_neurons, n_neurons, 2*max_lag+1))
        lag_matrix = np.zeros((n_neurons, n_neurons))
        
        movement_frame_indices = []
        if hasattr(self, 'movement_periods') and self.movement_periods:
            for bout_start, bout_end in self.movement_periods:
                extended_start = max(0, bout_start - 4)
                extended_end = min(self.imaging_times[-1], bout_end + 4)
                
                start_idx = np.searchsorted(self.imaging_times, extended_start)
                end_idx = np.searchsorted(self.imaging_times, extended_end)
                movement_frame_indices.extend(range(start_idx, end_idx))
        
        if not movement_frame_indices:
            movement_frame_indices = range(len(self.imaging_times))
        
        for i in range(n_neurons):
            neuron_i = self.combined_activity[reference_indices[i]]
            
            if neuron_i['type'].endswith('_only'):
                if data_type == 'F':
                    sig1 = neuron_i['F']
                elif data_type == 'Fneu':
                    sig1 = neuron_i['Fneu']
                elif data_type == 'dff':
                    sig1 = self.calculate_dff(neuron_i['F'], neuron_i['Fneu'])
                else:  # spks
                    sig1 = neuron_i['spks']
            else:  # overlap
                if data_type == 'F':
                    sig1 = neuron_i[f'F_{reference_channel}']
                elif data_type == 'Fneu':
                    sig1 = neuron_i[f'Fneu_{reference_channel}']
                elif data_type == 'dff':
                    sig1 = self.calculate_dff(neuron_i[f'F_{reference_channel}'], neuron_i[f'Fneu_{reference_channel}'])
                else:  # spks
                    sig1 = neuron_i[f'spks_{reference_channel}']
            
            sig1 = sig1[movement_frame_indices]
            sig1 = zscore(sig1)
            
            for j in range(i, n_neurons):
                neuron_j = self.combined_activity[reference_indices[j]]
                
                if neuron_j['type'].endswith('_only'):
                    if data_type == 'F':
                        sig2 = neuron_j['F']
                    elif data_type == 'Fneu':
                        sig2 = neuron_j['Fneu']
                    elif data_type == 'dff':
                        sig2 = self.calculate_dff(neuron_j['F'], neuron_j['Fneu'])
                    else:  # spks
                        sig2 = neuron_j['spks']
                else:  # overlap
                    if data_type == 'F':
                        sig2 = neuron_j[f'F_{reference_channel}']
                    elif data_type == 'Fneu':
                        sig2 = neuron_j[f'Fneu_{reference_channel}']
                    elif data_type == 'dff':
                        sig2 = self.calculate_dff(neuron_j[f'F_{reference_channel}'], neuron_j[f'Fneu_{reference_channel}'])
                    else:  # spks
                        sig2 = neuron_j[f'spks_{reference_channel}']
                
                sig2 = sig2[movement_frame_indices]
                sig2 = zscore(sig2)
                
                corr = signal.correlate(sig1, sig2, mode='full', method='auto')
                
                lags = signal.correlation_lags(len(sig1), len(sig2), mode='full')
                lag_idx = np.argmax(np.abs(corr))
                max_lag_value = lags[lag_idx]
                
                norm_factor = np.sqrt(np.sum(sig1**2) * np.sum(sig2**2))
                if norm_factor > 0:
                    corr = corr / norm_factor

                center = len(corr) // 2
                cross_corr_matrix[i, j] = corr[center-max_lag:center+max_lag+1]
                cross_corr_matrix[j, i] = cross_corr_matrix[i, j]
                
                lag_matrix[i, j] = max_lag_value
                lag_matrix[j, i] = -max_lag_value
        
        zero_lag_corr = cross_corr_matrix[:, :, max_lag]
        
        return reference_indices, zero_lag_corr, lag_matrix
    
    def plot_cross_relation_analysis(self, threshold=0.3, data_type='spks', reference_channel='ch1'):
        if not self.loaded:
            return None
        
        ch_indices, corr_matrix, lag_matrix = self.compute_cross_correlation_with_lags(data_type, reference_channel)
        
        if corr_matrix is None:
            return None
        
        fig = Figure(figsize=(15, 7), dpi=100)
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
        
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label('Cross-correlation at zero lag')
        ax1.set_title(f'Cross-correlation Matrix of {reference_channel} Neurons ({data_type})')
        ax1.set_xlabel('ROIs')
        ax1.set_ylabel('ROIs')
        
        ax2 = fig.add_subplot(gs[0, 1])
        G_directed = nx.DiGraph()
        
        for i, neuron_idx in enumerate(ch_indices):
            neuron = self.combined_neurons[neuron_idx]
            ypix, xpix = neuron['coords']
            center_y = np.mean(ypix)
            center_x = np.mean(xpix)
            
            G_directed.add_node(i, 
                              pos=(center_x, center_y),
                              neuron_idx=neuron_idx,
                              type=neuron['type'])
        
        n_neurons = len(ch_indices)
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and abs(corr_matrix[i, j]) > threshold:
                    if lag_matrix[i, j] > 0:
                        G_directed.add_edge(i, j, weight=corr_matrix[i, j], lag=lag_matrix[i, j])
                    elif lag_matrix[i, j] < 0:
                        G_directed.add_edge(j, i, weight=corr_matrix[i, j], lag=-lag_matrix[i, j])
        
        pos_directed = nx.get_node_attributes(G_directed, 'pos')
        
        edges_directed = G_directed.edges()
        weights_directed = [G_directed[u][v]['weight'] for u, v in edges_directed]
        lags_directed = [G_directed[u][v]['lag'] for u, v in edges_directed]
        
        edge_colors_directed = ['red' if w > 0 else 'blue' for w in weights_directed]
        edge_widths_directed = [abs(w) * 5 for w in weights_directed]
        
        nx.draw_networkx_edges(G_directed, pos_directed, edgelist=edges_directed, 
                              width=edge_widths_directed, edge_color=edge_colors_directed, 
                              alpha=0.7, ax=ax2, arrows=True, arrowsize=10)
        
        node_colors_directed = []
        for i in range(n_neurons):
            node_type = G_directed.nodes[i]['type']
            if node_type == 'overlap':
                node_colors_directed.append('yellow')
            else:
                channel_name = node_type.replace('_only', '')
                node_colors_directed.append(self.channel_colors.get(channel_name, 'gray'))
        
        nx.draw_networkx_nodes(G_directed, pos_directed, node_color=node_colors_directed, 
                              node_size=100, alpha=0.9, ax=ax2)
        
        labels_directed = {i: str(G_directed.nodes[i]['neuron_idx']) for i in range(n_neurons)}
        nx.draw_networkx_labels(G_directed, pos_directed, labels_directed, font_size=8, ax=ax2)
        
        ax2.set_xlim(0, self.Lx)
        ax2.set_ylim(self.Ly, 0)
        ax2.set_title(f'Directed Functional Network ({reference_channel}, {data_type}, Threshold: {threshold})')
        ax2.set_aspect('equal')
        
        legend_patches = []
        for channel_name in self.channels_data.keys():
            color = self.channel_colors.get(channel_name, 'gray')
            legend_patches.append(mpatches.Patch(color=color, label=f'{channel_name} Only'))
        
        if len(self.channels_data) > 1:
            legend_patches.append(mpatches.Patch(color='yellow', label='Overlap'))
        
        legend_patches.extend([
            mpatches.Patch(color='red', label='Positive Correlation'),
            mpatches.Patch(color='blue', label='Negative Correlation')
        ])
        
        ax2.legend(handles=legend_patches, loc='upper right')
        
        fig.tight_layout()
        return fig
    
    def plot_single_onset_analysis(self, onset_idx, pre_window=2, post_window=2):
        """Plot running and neuron dF/F for a single movement onset"""
        if not self.loaded or not self.movement_onsets or onset_idx >= len(self.movement_onsets):
            return None
        
        onset_time = self.movement_onsets[onset_idx]
        start_time = onset_time - pre_window
        end_time = onset_time + post_window
        
        # Get running data for the time window
        running_mask = (self.running_time_valid >= start_time) & (self.running_time_valid <= end_time)
        running_time_window = self.running_time_valid[running_mask] - onset_time
        running_data_window = self.running_data_valid[running_mask]
        
        # Get imaging data for the time window
        imaging_mask = (self.imaging_times >= start_time) & (self.imaging_times <= end_time)
        imaging_time_window = self.imaging_times[imaging_mask] - onset_time
        
        if not self.neuron_queue:
            return None
        
        # Create figure
        fig = Figure(figsize=(10, 8), dpi=100)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], figure=fig)
        
        # Plot running speed
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(running_time_window, running_data_window, 'b-', linewidth=2)
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=1, label='Onset')
        ax1.set_ylabel('Running Speed (cm/s)')
        ax1.set_title(f'Movement Onset {onset_idx} (Time: {onset_time:.2f}s)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot neuron dF/F
        ax2 = fig.add_subplot(gs[1])
        
        # Calculate dF/F for each neuron in queue
        neuron_dff_data = []
        for neuron_idx in self.neuron_queue:
            neuron = self.combined_activity[neuron_idx]
            
            if neuron['type'].endswith('_only'):
                dff = self.calculate_dff(neuron['F'], neuron['Fneu'])
            else:  # overlap
                first_channel = neuron['channels'][0]
                dff = self.calculate_dff(neuron[f'F_{first_channel}'], neuron[f'Fneu_{first_channel}'])
            
            dff_window = dff[imaging_mask]
            neuron_dff_data.append(dff_window)
        
        # Plot each neuron's dF/F with vertical offset
        y_offset = 0
        y_ticks = []
        y_tick_labels = []
        
        for i, neuron_idx in enumerate(self.neuron_queue):
            dff_window = neuron_dff_data[i]
            y_values = dff_window + y_offset
            ax2.plot(imaging_time_window, y_values, linewidth=1.5, label=f'Neuron {neuron_idx}')
            
            y_ticks.append(y_offset)
            y_tick_labels.append(f'{neuron_idx}')
            y_offset += 2  # Fixed interval between neurons
        
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=1)
        ax2.set_xlabel('Time from Onset (s)')
        ax2.set_ylabel('Neuron dF/F (offset)')
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(y_tick_labels)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        fig.tight_layout()
        return fig
    
    def plot_all_onsets_analysis(self, pre_window=2, post_window=2):
        """Plot average running and z-scored neuron activity for all movement onsets"""
        if not self.loaded or not self.movement_onsets or not self.neuron_queue:
            return None
        
        # Create time array for the analysis window
        time_from_onset = np.linspace(-pre_window, post_window, int((pre_window + post_window) * self.frame_rate) + 1)
        
        # Arrays to store data for each onset
        all_running_data = []
        all_neuron_zscore_data = []
        
        for onset_time in self.movement_onsets:
            start_time = onset_time - pre_window
            end_time = onset_time + post_window
            
            # Get running data for this onset
            running_mask = (self.running_time_valid >= start_time) & (self.running_time_valid <= end_time)
            running_time_onset = self.running_time_valid[running_mask] - onset_time
            running_data_onset = self.running_data_valid[running_mask]
            
            # Interpolate running data to common time grid
            if len(running_time_onset) > 1:
                running_interp = interp1d(running_time_onset, running_data_onset, 
                                         bounds_error=False, fill_value=0)
                running_interp_data = running_interp(time_from_onset)
                all_running_data.append(running_interp_data)
            
            # Get imaging data for this onset
            imaging_mask = (self.imaging_times >= start_time) & (self.imaging_times <= end_time)
            imaging_time_onset = self.imaging_times[imaging_mask] - onset_time
            
            # Calculate z-scored dF/F for each neuron
            neuron_zscore_onset = []
            for neuron_idx in self.neuron_queue:
                neuron = self.combined_activity[neuron_idx]
                
                if neuron['type'].endswith('_only'):
                    dff = self.calculate_dff(neuron['F'], neuron['Fneu'])
                else:  # overlap
                    first_channel = neuron['channels'][0]
                    dff = self.calculate_dff(neuron[f'F_{first_channel}'], neuron[f'Fneu_{first_channel}'])
                
                dff_onset = dff[imaging_mask]
                
                # Calculate baseline (pre-onset period)
                baseline_mask = imaging_time_onset < 0
                if np.any(baseline_mask):
                    baseline_mean = np.mean(dff_onset[baseline_mask])
                    baseline_std = np.std(dff_onset[baseline_mask])
                    
                    # Z-score using baseline period
                    if baseline_std > 0:
                        zscore_data = (dff_onset - baseline_mean) / baseline_std
                    else:
                        zscore_data = np.zeros_like(dff_onset)
                else:
                    zscore_data = np.zeros_like(dff_onset)
                
                # Interpolate to common time grid
                if len(imaging_time_onset) > 1:
                    zscore_interp = interp1d(imaging_time_onset, zscore_data, 
                                           bounds_error=False, fill_value=0)
                    zscore_interp_data = zscore_interp(time_from_onset)
                    neuron_zscore_onset.append(zscore_interp_data)
            
            if neuron_zscore_onset:
                all_neuron_zscore_data.append(np.array(neuron_zscore_onset))
        
        if not all_running_data or not all_neuron_zscore_data:
            return None
        
        # Convert to arrays
        all_running_data = np.array(all_running_data)
        all_neuron_zscore_data = np.array(all_neuron_zscore_data)  # shape: (n_onsets, n_neurons, n_timepoints)
        
        # Calculate mean and std across onsets
        running_mean = np.mean(all_running_data, axis=0)
        running_std = np.std(all_running_data, axis=0)
        
        neuron_zscore_mean = np.mean(all_neuron_zscore_data, axis=0)  # mean across onsets for each neuron
        neuron_zscore_std = np.std(all_neuron_zscore_data, axis=0)
        
        # Create figure with multiple subplots
        fig = Figure(figsize=(12, 10), dpi=100)
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
        
        # Plot 1: Running speed mean ± std
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_from_onset, running_mean, 'b-', linewidth=2, label='Mean')
        ax1.fill_between(time_from_onset, running_mean - running_std, running_mean + running_std, 
                        alpha=0.3, color='b', label='±Std')
        ax1.axvline(x=0, color='r', linestyle='--', linewidth=1, label='Onset')
        ax1.set_ylabel('Running Speed (cm/s)')
        ax1.set_title('Running Speed (Mean ± Std)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Neuron z-score mean ± std (averaged across neurons)
        ax2 = fig.add_subplot(gs[0, 1])
        overall_zscore_mean = np.mean(neuron_zscore_mean, axis=0)  # mean across neurons
        overall_zscore_std = np.std(neuron_zscore_mean, axis=0)    # std across neurons
        
        ax2.plot(time_from_onset, overall_zscore_mean, 'g-', linewidth=2, label='Mean')
        ax2.fill_between(time_from_onset, overall_zscore_mean - overall_zscore_std, 
                        overall_zscore_mean + overall_zscore_std, alpha=0.3, color='g', label='±Std')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=1, label='Onset')
        ax2.set_ylabel('Z-scored dF/F')
        ax2.set_title('Neuron Activity (Mean ± Std across neurons)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Running speed heatmap (onsets × time)
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(all_running_data, aspect='auto', cmap='viridis',
                        extent=[-pre_window, post_window, 0, len(all_running_data)],
                        origin='lower')
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=1)
        ax3.set_xlabel('Time from Onset (s)')
        ax3.set_ylabel('Onset Index')
        ax3.set_title('Running Speed Heatmap')
        fig.colorbar(im3, ax=ax3, label='Running Speed (cm/s)')
        
        # Plot 4: Neuron z-score heatmap (neurons × time, averaged across onsets)
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(neuron_zscore_mean, aspect='auto', cmap='RdBu_r',
                        extent=[-pre_window, post_window, 0, len(self.neuron_queue)],
                        origin='lower', vmin=-2, vmax=2)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=1)
        ax4.set_xlabel('Time from Onset (s)')
        ax4.set_ylabel('Neuron Index')
        ax4.set_yticks(range(len(self.neuron_queue)))
        ax4.set_yticklabels([str(idx) for idx in self.neuron_queue])
        ax4.set_title('Z-scored dF/F Heatmap (Mean across onsets)')
        fig.colorbar(im4, ax=ax4, label='Z-score')
        
        # Plot 5: Individual neuron traces (mean across onsets)
        ax5 = fig.add_subplot(gs[2, :])
        for i, neuron_idx in enumerate(self.neuron_queue):
            ax5.plot(time_from_onset, neuron_zscore_mean[i], 
                    label=f'Neuron {neuron_idx}', linewidth=1.5)
        
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=1, label='Onset')
        ax5.set_xlabel('Time from Onset (s)')
        ax5.set_ylabel('Z-scored dF/F')
        ax5.set_title('Individual Neuron Responses (Mean across onsets)')
        ax5.legend(ncol=min(5, len(self.neuron_queue)), loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        fig.suptitle(f'Movement Onset Analysis ({len(self.movement_onsets)} onsets, {len(self.neuron_queue)} neurons)', 
                    fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        return fig

    def plot_full_trace_aligned(self, spacing=2):
        """Plot full trace of running and neuron dF/F aligned with fixed spacing"""
        if not self.loaded or not self.neuron_queue:
            return None
        
        # Create figure
        fig = Figure(figsize=(12, 8), dpi=100)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], figure=fig)
        
        # Plot running speed (full trace)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.running_time_valid, self.running_data_valid, 'b-', linewidth=1.5, alpha=0.8)
        ax1.set_ylabel('Running Speed (cm/s)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.set_title('Full Trace: Running Speed and Neuron dF/F')
        ax1.grid(False)
        
        # Plot neuron dF/F (full trace) with vertical offset
        ax2 = fig.add_subplot(gs[1])
        
        # Calculate dF/F for each neuron in queue
        neuron_dff_data = []
        for neuron_idx in self.neuron_queue:
            neuron = self.combined_activity[neuron_idx]
            
            if neuron['type'].endswith('_only'):
                dff = self.calculate_dff(neuron['F'], neuron['Fneu'])
            else:  # overlap
                first_channel = neuron['channels'][0]
                dff = self.calculate_dff(neuron[f'F_{first_channel}'], neuron[f'Fneu_{first_channel}'])
            
            neuron_dff_data.append(dff)
        
        # Plot each neuron's dF/F with vertical offset
        y_offset = 0
        y_ticks = []
        y_tick_labels = []
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        for i, neuron_idx in enumerate(self.neuron_queue):
            dff = neuron_dff_data[i]
            color = colors[i % len(colors)]
            y_values = dff + y_offset
            ax2.plot(self.imaging_times, y_values, color=color, linewidth=1.2, 
                    label=f'Neuron {neuron_idx}', alpha=0.8)
            
            y_ticks.append(y_offset)
            y_tick_labels.append(f'{neuron_idx}')
            y_offset += spacing  # Fixed interval between neurons
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Neuron dF/F (offset)')
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(y_tick_labels)
        ax2.grid(False)
        
        # Set same x-axis limits for both subplots
        min_time = min(self.imaging_times[0], self.running_time_valid[0])
        max_time = max(self.imaging_times[-1], self.running_time_valid[-1])
        ax1.set_xlim(min_time, max_time)
        ax2.set_xlim(min_time, max_time)
        
        fig.tight_layout()
        return fig
    
class CalciumImagingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Calcium Imaging Analyzer")
        self.root.state("zoomed")
        self.analyzer = CalciumImagingAnalyzer()
        
        self.map_canvas = None
        self.map_toolbar = None
        self.activity_canvas = None
        self.activity_toolbar = None
        self.running_canvas = None
        self.running_toolbar = None
        
        self.data_type_var = tk.StringVar(value='dff')
        self.movement_types_var = {}
        self.log_text_widget = None
        
        # Create menu bar first
        self.create_menu_bar()
        
        # Create main display area
        self.create_widgets()
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Data", command=self.select_experiment_dir)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis Menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        
        # Running Data Submenu
        running_analysis_menu = tk.Menu(analysis_menu, tearoff=0)
        analysis_menu.add_cascade(label="Running Data Analysis", menu=running_analysis_menu)
        running_analysis_menu.add_command(label="Movement Periods", 
                                         command=lambda: self.running_data_analysis_wrapper('movement_periods'))
        running_analysis_menu.add_command(label="Rest Periods", 
                                         command=lambda: self.running_data_analysis_wrapper('rest_periods'))
        running_analysis_menu.add_command(label="General Onsets", 
                                         command=lambda: self.running_data_analysis_wrapper('general_onsets'))
        running_analysis_menu.add_command(label="Jerks", 
                                         command=lambda: self.running_data_analysis_wrapper('jerks'))
        running_analysis_menu.add_command(label="Locomotion Initiations", 
                                         command=lambda: self.running_data_analysis_wrapper('locomotion_initiations'))
        running_analysis_menu.add_command(label="Continuous Locomotion Periods", 
                                         command=lambda: self.running_data_analysis_wrapper('continuous_locomotion_periods'))
        running_analysis_menu.add_command(label="Locomotion Terminations", 
                                         command=lambda: self.running_data_analysis_wrapper('locomotion_terminations'))
        
        analysis_menu.add_separator()
        
        # Cross-relation Analysis
        analysis_menu.add_command(label="Cross-relation Analysis", 
                                 command=self.plot_cross_relation_analysis)
        
        analysis_menu.add_separator()
        
        # Movement Onset Analysis Submenu
        onset_analysis_menu = tk.Menu(analysis_menu, tearoff=0)
        analysis_menu.add_cascade(label="Movement Onset Analysis", menu=onset_analysis_menu)
        onset_analysis_menu.add_command(label="Single Onset", 
                                        command=lambda: self.open_onset_analysis_window('single'))
        onset_analysis_menu.add_command(label="All Onsets", 
                                        command=lambda: self.open_onset_analysis_window('all'))
        onset_analysis_menu.add_command(label="Full Trace", 
                                        command=lambda: self.plot_full_trace())
        
        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Data Type Submenu
        data_type_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Data Type", menu=data_type_menu)
        for dtype in ['F', 'Fneu', 'dff', 'spks']:
            data_type_menu.add_radiobutton(label=dtype, variable=self.data_type_var, 
                                          value=dtype, command=self.on_data_type_change)
        
        view_menu.add_separator()
        
        # Movement Types Submenu
        movement_types_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Movement Types", menu=movement_types_menu)
        
        movement_types = [
            ('Movement Periods', 'movement_periods'),
            ('Rest Periods', 'rest_periods'),
            ('Movement Onsets', 'movement_onsets'),
            ('Jerks', 'jerks'),
            ('Locomotion Initiations', 'locomotion_initiations'),
            ('Continuous Locomotion', 'continuous_locomotion_periods'),
            ('Locomotion Terminations', 'locomotion_terminations')
        ]
        
        for display_name, var_name in movement_types:
            var = tk.BooleanVar(value=False)
            self.movement_types_var[var_name] = var
            movement_types_menu.add_checkbutton(label=display_name, variable=var,
                                               command=self.on_movement_types_change)
        
        # Settings Menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Network Threshold", command=self.set_network_threshold)
        settings_menu.add_command(label="Reference Channel", command=self.set_reference_channel)
    
    def setup_log_display(self):
        for widget in self.log_frame.winfo_children():
            widget.destroy()
        
        self.log_text_widget = tk.Text(self.log_frame, wrap=tk.WORD, height=25, font=("Consolas", 9))
        self.log_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        set_log_widget(self.log_text_widget)
        log_message("The log system has been initialized. All messages will be displayed here.", "INFO")
        
    def create_widgets(self):
        """Create main display widgets"""
        display_frame = tk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        display_frame.grid_rowconfigure(0, weight=0)
        display_frame.grid_rowconfigure(1, weight=0)
        display_frame.grid_columnconfigure(0, weight=0)
        display_frame.grid_columnconfigure(1, weight=0)
        
        # Neuron map frame (left)
        map_frame = tk.Frame(display_frame, width=1024, height=1024, bg="#e0e0e0")
        map_frame.grid(row=0, column=0, rowspan=1, sticky='nw', padx=0, pady=0)
        map_frame.grid_propagate(False)
        
        right_frame = tk.Frame(display_frame)
        right_frame.grid(row=0, column=1, rowspan=3, sticky='nw', padx=0, pady=0)
        right_frame.grid_rowconfigure(0, weight=0)
        right_frame.grid_rowconfigure(1, weight=0)
        right_frame.grid_rowconfigure(2, weight=1)
        right_frame.grid_columnconfigure(0, weight=0)

        # Activity plot frame (right1)
        activity_frame = tk.Frame(right_frame, width=900, height=300, bg="#e0e0e0")
        activity_frame.grid(row=0, column=0, sticky='nw', padx=0, pady=0)
        activity_frame.grid_propagate(False)
        
        # Running plot frame (right2)
        running_frame = tk.Frame(right_frame, width=900, height=300, bg="#e0e0e0")
        running_frame.grid(row=1, column=0, sticky='nw', padx=0, pady=0)
        running_frame.grid_propagate(False)

        # Control and log frame (right3)
        self.log_frame = tk.Frame(right_frame, width=900, height=300, bg="#e0e0e0")
        self.log_frame.grid(row=2, column=0, sticky='nsew', padx=0, pady=0)
        
        self.setup_log_display()

        self.map_frame = map_frame
        self.activity_frame = activity_frame
        self.running_frame = running_frame
        
        # Status bar
        self.status_var = tk.StringVar()
        log_message("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_experiment_dir(self):
        """Select experiment directory"""
        path = filedialog.askdirectory(title="Select Experiment Directory")
        if path:
            self.analyzer.base_dir = path
            log_message(f"Experiment Dir: {os.path.basename(path)}")
            self.load_data()
    
    def load_data(self):
        """Load data from selected directory"""
        if not self.analyzer.base_dir:
            log_message("Please select an experiment directory first", "ERROR")
            return
        
        try:
            log_message("Loading data...")
            self.root.update()
            
            success, message = self.analyzer.load_from_directory(self.analyzer.base_dir)
            
            if success:
                log_message("Data loaded successfully!")
                
                # # Initialize neuron queue with all neurons
                # self.analyzer.neuron_queue = list(range(len(self.analyzer.combined_neurons)))
                
                # Plot initial visualizations
                self.plot_neuron_map()
                self.plot_activity()
                self.plot_running()
            else:
                log_message(message, "ERROR")
                
        except Exception as e:
            log_message(str(e), "ERROR")
    
    def on_data_type_change(self, *args):
        """Handle data type change"""
        self.plot_activity()
    
    def on_movement_types_change(self):
        """Handle movement types change"""
        self.plot_activity()
        self.plot_running()
    
    def get_selected_movement_types(self):
        """Get list of selected movement types"""
        selected_types = []
        for var_name, var in self.movement_types_var.items():
            if var.get():
                selected_types.append(var_name)
        return selected_types
    
    def plot_neuron_map(self):
        """Plot neuron map"""
        if not self.analyzer.loaded:
            return
        
        if self.map_canvas:
            self.map_canvas.get_tk_widget().destroy()
            if self.map_toolbar:
                self.map_toolbar.destroy()
        
        fig = self.analyzer.plot_neuron_mask(
            self.analyzer.selected_neuron_idx, 
            self.analyzer.neuron_queue
        )
        
        self.map_canvas = FigureCanvasTkAgg(fig, master=self.map_frame)
        self.map_canvas.draw()
        
        self.map_canvas.mpl_connect('button_press_event', self.on_map_click)
        
        self.map_toolbar = NavigationToolbar2Tk(self.map_canvas, self.map_frame)
        self.map_toolbar.update()
        self.map_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_activity(self):
        """Plot activity"""
        if not self.analyzer.loaded:
            return
        
        if self.activity_canvas:
            self.activity_canvas.get_tk_widget().destroy()
            if self.activity_toolbar:
                self.activity_toolbar.destroy()
        
        data_type = self.data_type_var.get()
        movement_types = self.get_selected_movement_types()
        
        if len(self.analyzer.neuron_queue) == 1:
            fig = self.analyzer.plot_neuron_activity(self.analyzer.neuron_queue[0], data_type, movement_types)
        else:
            fig = self.analyzer.plot_multiple_neurons_activity(self.analyzer.neuron_queue, data_type, movement_types)
        
        if fig:
            self.activity_canvas = FigureCanvasTkAgg(fig, master=self.activity_frame)
            self.activity_canvas.draw()
            self.activity_toolbar = NavigationToolbar2Tk(self.activity_canvas, self.activity_frame)
            self.activity_toolbar.update()
            self.activity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_running(self):
        """Plot running activity"""
        if not self.analyzer.loaded:
            return
        
        if self.running_canvas:
            self.running_canvas.get_tk_widget().destroy()
            if self.running_toolbar:
                self.running_toolbar.destroy()
        
        movement_types = self.get_selected_movement_types()
        fig = self.analyzer.plot_running_activity(movement_types)
        
        if fig:
            self.running_canvas = FigureCanvasTkAgg(fig, master=self.running_frame)
            self.running_canvas.draw()
            self.running_toolbar = NavigationToolbar2Tk(self.running_canvas, self.running_frame)
            self.running_toolbar.update()
            self.running_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_map_click(self, event):
        """Handle map click events"""
        if not self.analyzer.loaded or event.inaxes is None:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        if 0 <= x < self.analyzer.Lx and 0 <= y < self.analyzer.Ly:
            neuron_id = self.analyzer.combined_mask[y, x]
            if neuron_id > 0:
                neuron_idx = neuron_id - 1
                
                if event.button == 1:  # Left click
                    self.analyzer.selected_neuron_idx = neuron_idx
                    
                    if neuron_idx not in self.analyzer.neuron_queue:
                        self.analyzer.neuron_queue.append(neuron_idx)
                    
                    log_message(f"Selected neuron {neuron_idx}")
                
                elif event.button == 3:  # Right click
                    if neuron_idx in self.analyzer.neuron_queue:
                        self.analyzer.neuron_queue.remove(neuron_idx)
                        
                        if self.analyzer.selected_neuron_idx == neuron_idx:
                            if self.analyzer.neuron_queue:
                                self.analyzer.selected_neuron_idx = self.analyzer.neuron_queue[0]
                            else:
                                self.analyzer.selected_neuron_idx = -1
                        
                        log_message(f"Removed neuron {neuron_idx} from queue")
                    else:
                        self.analyzer.neuron_queue.append(neuron_idx)
                        log_message(f"Added neuron {neuron_idx} to queue")
                
                self.plot_neuron_map()
                self.plot_activity()
    
    def running_data_analysis_wrapper(self, analysis_type):
        """Wrapper for running data analysis"""
        if not self.analyzer.loaded:
            log_message("Please load data first", "WARNING")
            return
        
        # Perform the analysis using the analyzer's existing method
        # This would need to be implemented in the CalciumImagingAnalyzer class
        log_message(f"Analyzing {analysis_type}...")
        
        # Update plots with analysis results
        self.on_movement_types_change()
        
        log_message(f"{analysis_type} analysis completed")
    
    def plot_cross_relation_analysis(self):
        """Plot cross-relation analysis"""
        if not self.analyzer.loaded:
            log_message("Please load data first", "WARNING")
            return
        
        # Open settings dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Cross-relation Analysis Settings")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Network Threshold:").grid(row=0, column=0, padx=10, pady=10)
        threshold_var = tk.DoubleVar(value=0.3)
        ttk.Entry(dialog, textvariable=threshold_var, width=10).grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(dialog, text="Data Type:").grid(row=1, column=0, padx=10, pady=10)
        dtype_var = tk.StringVar(value=self.data_type_var.get())
        ttk.Combobox(dialog, textvariable=dtype_var, values=['F', 'Fneu', 'dff', 'spks'],
                    state="readonly", width=8).grid(row=1, column=1, padx=10, pady=10)
        
        def plot_analysis():
            threshold = threshold_var.get()
            data_type = dtype_var.get()
            
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title(f"Cross-relation Analysis ({data_type})")
            analysis_window.geometry("1500x800")
            
            # Get reference channel (default to first channel)
            reference_channel = list(self.analyzer.channels_data.keys())[0]
            
            fig = self.analyzer.plot_cross_relation_analysis(
                threshold=threshold, 
                data_type=data_type, 
                reference_channel=reference_channel
            )
            
            if fig:
                canvas = FigureCanvasTkAgg(fig, master=analysis_window)
                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, analysis_window)
                toolbar.update()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            dialog.destroy()
        
        ttk.Button(dialog, text="Plot", command=plot_analysis).grid(row=2, column=0, columnspan=2, pady=10)
    
    def open_onset_analysis_window(self, plot_type):
        """Open onset analysis window"""
        if not self.analyzer.loaded:
            log_message("Please load data first", "WARNING")
            return
        
        if not hasattr(self.analyzer, 'movement_onsets') or not self.analyzer.movement_onsets:
            log_message("No movement onsets detected. Please run movement analysis first.", "WARNING")
            return
        
        if plot_type == 'single':
            # Open dialog for single onset selection
            dialog = tk.Toplevel(self.root)
            dialog.title("Single Onset Analysis")
            dialog.geometry("350x200")
            dialog.transient(self.root)
            dialog.grab_set()
            
            ttk.Label(dialog, text="Select Onset:").grid(row=0, column=0, padx=10, pady=10)
            onset_var = tk.StringVar()
            onset_options = [f"{i} (t={t:.2f}s)" for i, t in enumerate(self.analyzer.movement_onsets)]
            onset_combo = ttk.Combobox(dialog, textvariable=onset_var, values=onset_options, 
                                       state="readonly", width=20)
            onset_combo.grid(row=0, column=1, padx=10, pady=10)
            if onset_options:
                onset_combo.current(0)
            
            ttk.Label(dialog, text="Pre-window (s):").grid(row=1, column=0, padx=10, pady=10)
            pre_var = tk.DoubleVar(value=2.0)
            ttk.Entry(dialog, textvariable=pre_var, width=10).grid(row=1, column=1, padx=10, pady=10)
            
            ttk.Label(dialog, text="Post-window (s):").grid(row=2, column=0, padx=10, pady=10)
            post_var = tk.DoubleVar(value=2.0)
            ttk.Entry(dialog, textvariable=post_var, width=10).grid(row=2, column=1, padx=10, pady=10)
            
            def plot_single():
                onset_idx = int(onset_var.get().split()[0])
                fig = self.analyzer.plot_single_onset_analysis(onset_idx, pre_var.get(), post_var.get())
                
                if fig:
                    plot_window = tk.Toplevel(self.root)
                    plot_window.title(f"Single Onset Analysis - Onset {onset_idx}")
                    plot_window.geometry("1200x900")
                    
                    canvas = FigureCanvasTkAgg(fig, master=plot_window)
                    canvas.draw()
                    toolbar = NavigationToolbar2Tk(canvas, plot_window)
                    toolbar.update()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                dialog.destroy()
            
            ttk.Button(dialog, text="Plot", command=plot_single).grid(row=3, column=0, columnspan=2, pady=10)
        
        elif plot_type == 'all':
            # Open dialog for all onsets analysis
            dialog = tk.Toplevel(self.root)
            dialog.title("All Onsets Analysis")
            dialog.geometry("300x150")
            dialog.transient(self.root)
            dialog.grab_set()
            
            ttk.Label(dialog, text="Pre-window (s):").grid(row=0, column=0, padx=10, pady=10)
            pre_var = tk.DoubleVar(value=2.0)
            ttk.Entry(dialog, textvariable=pre_var, width=10).grid(row=0, column=1, padx=10, pady=10)
            
            ttk.Label(dialog, text="Post-window (s):").grid(row=1, column=0, padx=10, pady=10)
            post_var = tk.DoubleVar(value=2.0)
            ttk.Entry(dialog, textvariable=post_var, width=10).grid(row=1, column=1, padx=10, pady=10)
            
            def plot_all():
                fig = self.analyzer.plot_all_onsets_analysis(pre_var.get(), post_var.get())
                
                if fig:
                    plot_window = tk.Toplevel(self.root)
                    plot_window.title("All Onsets Analysis")
                    plot_window.geometry("1200x900")
                    
                    canvas = FigureCanvasTkAgg(fig, master=plot_window)
                    canvas.draw()
                    toolbar = NavigationToolbar2Tk(canvas, plot_window)
                    toolbar.update()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                dialog.destroy()
            
            ttk.Button(dialog, text="Plot", command=plot_all).grid(row=2, column=0, columnspan=2, pady=10)
    
    def plot_full_trace(self):
        """Plot full trace"""
        if not self.analyzer.loaded:
            log_message("Please load data first", "WARNING")
            return
        
        # Open dialog for spacing setting
        dialog = tk.Toplevel(self.root)
        dialog.title("Full Trace Settings")
        dialog.geometry("300x100")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Neuron Spacing:").grid(row=0, column=0, padx=10, pady=10)
        spacing_var = tk.DoubleVar(value=2.0)
        ttk.Entry(dialog, textvariable=spacing_var, width=10).grid(row=0, column=1, padx=10, pady=10)
        
        def plot_trace():
            fig = self.analyzer.plot_full_trace_aligned(spacing_var.get())
            
            if fig:
                plot_window = tk.Toplevel(self.root)
                plot_window.title("Full Trace Analysis")
                plot_window.geometry("1200x900")
                
                canvas = FigureCanvasTkAgg(fig, master=plot_window)
                canvas.draw()
                toolbar = NavigationToolbar2Tk(canvas, plot_window)
                toolbar.update()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            dialog.destroy()
        
        ttk.Button(dialog, text="Plot", command=plot_trace).grid(row=1, column=0, columnspan=2, pady=10)
    
    def set_network_threshold(self):
        """Set network threshold"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Network Threshold")
        dialog.geometry("250x100")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Threshold:").grid(row=0, column=0, padx=10, pady=10)
        threshold_var = tk.DoubleVar(value=0.3)
        ttk.Entry(dialog, textvariable=threshold_var, width=10).grid(row=0, column=1, padx=10, pady=10)
        
        def apply():
            # Store threshold for later use
            self.network_threshold = threshold_var.get()
            dialog.destroy()
        
        ttk.Button(dialog, text="Apply", command=apply).grid(row=1, column=0, columnspan=2, pady=10)
    
    def set_reference_channel(self):
        """Set reference channel"""
        if not self.analyzer.loaded:
            log_message("Please load data first", "WARNING")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Reference Channel")
        dialog.geometry("250x100")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Channel:").grid(row=0, column=0, padx=10, pady=10)
        channel_var = tk.StringVar(value=list(self.analyzer.channels_data.keys())[0])
        ttk.Combobox(dialog, textvariable=channel_var, 
                    values=list(self.analyzer.channels_data.keys()),
                    state="readonly", width=10).grid(row=0, column=1, padx=10, pady=10)
        
        def apply():
            # Store reference channel for later use
            self.reference_channel = channel_var.get()
            dialog.destroy()
        
        ttk.Button(dialog, text="Apply", command=apply).grid(row=1, column=0, columnspan=2, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = CalciumImagingApp(root)
    root.mainloop()