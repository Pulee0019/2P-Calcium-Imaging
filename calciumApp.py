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
        
        self.channel_colors = {
            'ch1': 'green',
            'ch2': 'red',
            'ch3': 'blue',
            'ch4': 'yellow',
            'ch5': 'cyan',
            'ch6': 'magenta'
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
        
        running_files = glob.glob(os.path.join(base_directory, '**', '*running*.dat*'), recursive=True)
        if running_files:
            dat2_files = [f for f in running_files if f.endswith('.dat2')]
            if dat2_files:
                result['running_path'] = dat2_files[0]
            else:
                result['running_path'] = running_files[0]
        
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
            
            first_channel = list(self.channels_data.values())[0]
            self.Ly = first_channel['ops']['Ly']
            self.Lx = first_channel['ops']['Lx']
            
            for channel_name, data in self.channels_data.items():
                if data['ops']['Ly'] != self.Ly or data['ops']['Lx'] != self.Lx:
                    return False, f"Channel {channel_name} dimensions don't match!"
            
            data = np.fromfile(self.running_path, dtype=np.float64)
            data = data.reshape(-1, 100, 5)
            self.running_data = -data[:, :, 1].ravel() * 7.5 / 360 * np.pi
            raw_running_time = data[:, :, 0].ravel()
            self.running_relative_time = (raw_running_time - raw_running_time[0]) * 24 * 60
            
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
                print("LSM cannot find!")
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
                print("Warning: Trigger signals not found, using zero offset")
            
            self.running_time = self.running_relative_time * 60 - self.relative_time
            
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
            
            self.detect_movement_bouts()
            
            self.loaded = True
            print("Data loaded successfully!")
            print(f"Total channels: {len(self.channels_data)}")
            print(f"Total neurons: {len(self.combined_neurons)}")
            
            for channel_name in self.channels_data.keys():
                channel_only = len([n for n in self.combined_neurons if n['type'] == f'{channel_name}_only'])
                print(f"{channel_name} only: {channel_only}")
            
            if len(self.channels_data) > 1:
                print(f"Overlap: {len(self.overlap_pairs)}")
            
            print(f"Movement bouts detected: {len(self.movement_bouts)}")
            
            self.neuron_queue = [0]
            self.selected_neuron_idx = 0
            
            return True, "Data loaded successfully!"
            
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def detect_movement_bouts(self):
        """Detect movement bouts based on the provided criteria"""
        IMMOBILITY_THRESHOLD = 0.2  # cm/s
        MIN_IMMOBILITY_DURATION = 4  # seconds
        MIN_MOVEMENT_DURATION = 0.5    # seconds
        BUFFER_TIME = 0.1           # seconds
        
        if len(self.running_time_valid) > 1:
            sample_interval = np.mean(np.diff(self.running_time_valid))
            sample_rate = 1.0 / sample_interval
        else:
            sample_rate = 20.0
        
        min_immobility_samples = int(MIN_IMMOBILITY_DURATION * sample_rate)
        buffer_samples = int(BUFFER_TIME * sample_rate)
        
        immobility_mask = np.abs(self.running_data_valid) < IMMOBILITY_THRESHOLD
        
        immobility_regions = self.find_contiguous_regions(immobility_mask, min_immobility_samples)
        
        movement_regions = []
        
        if not immobility_regions:
            movement_regions.append((0, len(self.running_time_valid) - 1))
        else:
            if immobility_regions[0][0] > 0:
                movement_regions.append((0, immobility_regions[0][0] - 1))
            
            for i in range(len(immobility_regions) - 1):
                start = immobility_regions[i][1] + 1
                end = immobility_regions[i + 1][0] - 1
                duration = (end - start + 1) * sample_interval
                if start <= end and duration >= MIN_MOVEMENT_DURATION:
                    movement_regions.append((start, end))
            
            if immobility_regions[-1][1] < len(self.running_time_valid) - 1:
                movement_regions.append((immobility_regions[-1][1] + 1, len(self.running_time_valid) - 1))
        
        movement_bouts = []

        total_samples = len(self.running_data_valid)
        data_end_time = self.running_time_valid[-1]

        for movement_start, movement_end in movement_regions:
            prev_immobility = False
            for imm_start, imm_end in immobility_regions:
                if imm_end <= movement_start and (movement_start - imm_end) * sample_interval <= BUFFER_TIME:
                    immobility_duration = (imm_end - imm_start) * sample_interval
                    if immobility_duration >= MIN_IMMOBILITY_DURATION:
                        prev_immobility = True
                        break
            
            next_immobility = False
            remaining_time = data_end_time - self.running_time_valid[movement_end]
            
            if remaining_time < MIN_IMMOBILITY_DURATION:
                next_immobility = True
            else:
                for imm_start, imm_end in immobility_regions:
                    if imm_start >= movement_end and (imm_start - movement_end) * sample_interval <= BUFFER_TIME:
                        immobility_duration = (imm_end - imm_start) * sample_interval
                        if immobility_duration >= MIN_IMMOBILITY_DURATION:
                            next_immobility = True
                            break
            
            if prev_immobility and (next_immobility or remaining_time < MIN_IMMOBILITY_DURATION):
                immobility_std = np.std(self.running_data_valid[immobility_mask])
                movement_std_threshold = 2 * immobility_std
                
                onset_idx = movement_start
                for i in range(max(0, movement_start - buffer_samples), movement_start):
                    if np.abs(self.running_data_valid[i]) > movement_std_threshold:
                        onset_idx = i
                        break
                
                offset_idx = movement_end
                search_end = min(total_samples - 1, movement_end + buffer_samples)
                for i in range(search_end, movement_end, -1):
                    if np.abs(self.running_data_valid[i]) > movement_std_threshold:
                        offset_idx = i
                        break
                
                movement_bouts.append((
                    self.running_time_valid[onset_idx],
                    self.running_time_valid[offset_idx]
                ))

        self.movement_bouts = movement_bouts
        return movement_bouts
    
    def find_contiguous_regions(self, mask, min_samples):
        """Find contiguous regions in a boolean mask with minimum length"""
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
        
        if in_region and len(mask) - region_start >= min_samples:
            regions.append((region_start, len(mask)))
        
        return regions
    
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
    
    def plot_neuron_activity(self, neuron_idx, data_type='dff'):
        if not self.loaded or neuron_idx < 0 or neuron_idx >= len(self.combined_activity):
            return None
        
        neuron = self.combined_activity[neuron_idx]
        neuron_type = neuron['type']
        
        fig = Figure(figsize=(12, 3), dpi=100)
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
        
        if hasattr(self, 'movement_bouts'):
            for bout_start, bout_end in self.movement_bouts:
                ax.axvspan(bout_start, bout_end, alpha=0.2, color='blue')
        
        ax.set_title(f'Neuron {neuron_idx} Activity ({neuron_type}) - {data_type}', fontsize=14)
        ax.grid(False)
        ax.legend(loc='upper right')
        
        min_time = min(self.imaging_times[0], self.running_time_valid[0])
        max_time = max(self.imaging_times[-1], self.running_time_valid[-1])
        ax.set_xlim(min_time, max_time)
        
        fig.tight_layout()
        return fig
    
    def plot_multiple_neurons_activity(self, neuron_indices, data_type='dff'):
        if not self.loaded or not neuron_indices:
            return None
        
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        fig = Figure(figsize=(12, 3), dpi=100)
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
                
                ax.plot(self.imaging_times, data, color=color, 
                       label=f'Neuron {neuron_idx} (overlap)', alpha=0.8)
        
        if hasattr(self, 'movement_bouts'):
            for bout_start, bout_end in self.movement_bouts:
                ax.axvspan(bout_start, bout_end, alpha=0.2, color='blue')
        
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
        ax.legend(loc='upper right')
        
        min_time = min(self.imaging_times[0], self.running_time_valid[0])
        max_time = max(self.imaging_times[-1], self.running_time_valid[-1])
        ax.set_xlim(min_time, max_time)
        
        fig.tight_layout()
        return fig
    
    def plot_running_activity(self):
        if not self.loaded:
            return None
        
        fig = Figure(figsize=(12, 2), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(self.running_time_valid, self.running_data_valid, 'b-', alpha=0.7, label='Running Speed')
        ax.set_ylabel('Running Speed(cm/s)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        if hasattr(self, 'movement_bouts'):
            for i, (bout_start, bout_end) in enumerate(self.movement_bouts):
                ax.axvspan(bout_start, bout_end, alpha=0.2, color='blue')
                if i == 0:
                    ax.axvspan(bout_start, bout_end, alpha=0.2, color='blue', label='Movement Bout')
        
        min_time = min(self.imaging_times[0], self.running_time_valid[0])
        max_time = max(self.imaging_times[-1], self.running_time_valid[-1])
        ax.set_xlim(min_time, max_time)
        ax.set_xlabel('Time (s)')
        
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        return fig
    
    def plot_neuron_mask(self, highlight_idx=-1, queue_indices=[]):
        if not self.loaded:
            return None
        
        fig = Figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        img = ax.imshow(self.combined_mask, cmap=self.cmap, 
                        vmin=0, vmax=len(self.combined_neurons), 
                        interpolation='nearest')
        
        if 0 <= highlight_idx < len(self.combined_neurons):
            neuron = self.combined_neurons[highlight_idx]
            ypix, xpix = neuron['coords']
            highlight_mask = np.zeros_like(self.combined_mask)
            highlight_mask[ypix, xpix] = 1
            ax.imshow(highlight_mask, cmap=ListedColormap(['none', 'cyan']), 
                     alpha=0.7, interpolation='nearest')
        
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
        
        ax.legend(handles=legend_patches, loc='upper right')
        
        ax.set_title(f'Multi-Channel Neuron Map ({len(self.combined_neurons)} neurons)', fontsize=14)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        fig.tight_layout()
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
        if hasattr(self, 'movement_bouts') and self.movement_bouts:
            for bout_start, bout_end in self.movement_bouts:
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
    
class CalciumImagingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Channel Calcium Imaging Analyzer")
        self.root.geometry("2000x900")
        self.analyzer = CalciumImagingAnalyzer()
        
        self.map_canvas = None
        self.map_toolbar = None
        self.activity_canvas = None
        self.activity_toolbar = None
        self.running_canvas = None
        self.running_toolbar = None
        
        self.data_type_var = tk.StringVar(value='dff')
        
        self.create_widgets()
    
    def create_widgets(self):
        control_frame = tk.Frame(self.root, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        display_frame = tk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_columnconfigure(1, weight=2)
        
        map_frame = tk.Frame(display_frame)
        map_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5, pady=5)
        
        activity_frame = tk.Frame(display_frame)
        activity_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        running_frame = tk.Frame(display_frame)
        running_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        self.map_frame = map_frame
        self.activity_frame = activity_frame
        self.running_frame = running_frame
        
        tk.Label(control_frame, text="Data Loading", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        
        dir_frame = tk.Frame(control_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(dir_frame, text="Select Experiment Directory", 
                 command=self.select_experiment_dir).pack(fill=tk.X, pady=2)
        
        tk.Button(control_frame, text="Load Data", command=self.load_data,
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(pady=10, fill=tk.X)
        
        tk.Label(control_frame, text="Data Type", font=("Arial", 12, "bold")).pack(pady=(20, 5), anchor="w")
        
        data_type_frame = tk.Frame(control_frame)
        data_type_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(data_type_frame, text="Data Type:").pack(side=tk.LEFT)
        data_type_combo = ttk.Combobox(data_type_frame, textvariable=self.data_type_var, 
                                      values=['F', 'Fneu', 'dff', 'spks'], state="readonly", width=10)
        data_type_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        data_type_combo.bind("<<ComboboxSelected>>", self.on_data_type_change)
        
        tk.Label(control_frame, text="Neuron Selection", font=("Arial", 12, "bold")).pack(pady=(20, 5), anchor="w")
        
        self.neuron_var = tk.StringVar()
        neuron_frame = tk.Frame(control_frame)
        neuron_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(neuron_frame, text="Select Neuron:").pack(side=tk.LEFT)
        self.neuron_combo = ttk.Combobox(neuron_frame, textvariable=self.neuron_var, state="disabled", width=5)
        self.neuron_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.neuron_combo.bind("<<ComboboxSelected>>", self.on_neuron_select)
        
        tk.Label(control_frame, text="Neuron Queue", font=("Arial", 12, "bold")).pack(pady=(20, 5), anchor="w")
        
        self.queue_listbox = tk.Listbox(control_frame, height=8)
        self.queue_listbox.pack(fill=tk.X, pady=5)
        self.queue_listbox.bind('<Button-1>', self.on_queue_select)
        self.queue_listbox.bind('<Button-3>', self.on_queue_remove)
        
        tk.Label(control_frame, text="Network Analysis", font=("Arial", 12, "bold")).pack(pady=(20, 5), anchor="w")
        
        network_frame = tk.Frame(control_frame)
        network_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(network_frame, text="Cross-relation Analysis", 
                 command=self.plot_cross_relation_analysis).pack(fill=tk.X, pady=2)
        
        threshold_frame = tk.Frame(control_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(threshold_frame, text="Network Threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.3)
        threshold_spinbox = tk.Spinbox(threshold_frame, from_=0.0, to=1.0, increment=0.05, 
                                      textvariable=self.threshold_var, width=5)
        threshold_spinbox.pack(side=tk.RIGHT)
        
        reference_frame = tk.Frame(control_frame)
        reference_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(reference_frame, text="Reference Channel:").pack(side=tk.LEFT)
        self.reference_var = tk.StringVar(value='ch1')
        reference_combo = ttk.Combobox(reference_frame, textvariable=self.reference_var, 
                                      values=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6'], 
                                      state="readonly", width=5)
        reference_combo.pack(side=tk.RIGHT)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        tk.Label(control_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, width=20).pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def select_experiment_dir(self):
        path = filedialog.askdirectory(title="Select Experiment Directory")
        if path:
            self.analyzer.base_dir = path
            self.status_var.set(f"Experiment Dir: {os.path.basename(path)}")
    
    def load_data(self):
        if not self.analyzer.base_dir:
            self.status_var.set("Error: Please select an experiment directory first")
            return
        
        try:
            self.status_var.set("Loading data...")
            self.root.update()
            
            success, message = self.analyzer.load_from_directory(self.analyzer.base_dir)
            
            if success:
                self.status_var.set("Data loaded successfully!")
                
                neuron_ids = [str(i) for i in range(len(self.analyzer.combined_neurons))]
                self.neuron_combo['values'] = neuron_ids
                self.neuron_combo['state'] = 'readonly'
                self.neuron_combo.current(0)
                
                channel_names = list(self.analyzer.channels_data.keys())
                self.reference_var.set(channel_names[0])
                
                self.update_queue_listbox()
                
                self.plot_neuron_map()
                self.plot_activity()
                self.plot_running()
            else:
                self.status_var.set(f"Error: {message}")
                messagebox.showerror("Error", message)
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def on_data_type_change(self, event):
        self.plot_activity()
    
    def update_queue_listbox(self):
        self.queue_listbox.delete(0, tk.END)
        for neuron_idx in self.analyzer.neuron_queue:
            self.queue_listbox.insert(tk.END, f"Neuron {neuron_idx}")
    
    def on_neuron_select(self, event):
        neuron_idx = int(self.neuron_var.get())
        self.analyzer.selected_neuron_idx = neuron_idx
        
        if neuron_idx not in self.analyzer.neuron_queue:
            self.analyzer.neuron_queue.append(neuron_idx)
            self.update_queue_listbox()
        
        self.status_var.set(f"Selected neuron {neuron_idx}")
        self.plot_neuron_map()
        self.plot_activity()
    
    def on_queue_select(self, event):
        selection = self.queue_listbox.curselection()
        if selection:
            index = selection[0]
            neuron_idx = self.analyzer.neuron_queue[index]
            self.analyzer.selected_neuron_idx = neuron_idx
            self.neuron_var.set(str(neuron_idx))
            self.status_var.set(f"Selected neuron {neuron_idx} from queue")
            self.plot_neuron_map()
            self.plot_activity()
    
    def on_queue_remove(self, event):
        selection = self.queue_listbox.curselection()
        if selection:
            index = selection[0]
            neuron_idx = self.analyzer.neuron_queue[index]
            
            self.analyzer.neuron_queue.pop(index)
            
            if self.analyzer.selected_neuron_idx == neuron_idx:
                if self.analyzer.neuron_queue:
                    self.analyzer.selected_neuron_idx = self.analyzer.neuron_queue[0]
                    self.neuron_var.set(str(self.analyzer.neuron_queue[0]))
                else:
                    self.analyzer.selected_neuron_idx = -1
                    self.neuron_var.set('')
            
            self.update_queue_listbox()
            self.status_var.set(f"Removed neuron {neuron_idx} from queue")
            self.plot_neuron_map()
            self.plot_activity()
    
    def plot_neuron_map(self):
        if not self.analyzer.loaded:
            self.status_var.set("Error: Load data first")
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
        
        self.status_var.set(f"Displaying neuron map. Total neurons: {len(self.analyzer.combined_neurons)}")
    
    def plot_activity(self):
        if not self.analyzer.loaded:
            self.status_var.set("Error: Load data first")
            return
        
        if self.activity_canvas:
            self.activity_canvas.get_tk_widget().destroy()
            if self.activity_toolbar:
                self.activity_toolbar.destroy()
        
        data_type = self.data_type_var.get()
        
        if len(self.analyzer.neuron_queue) == 1:
            fig = self.analyzer.plot_neuron_activity(self.analyzer.neuron_queue[0], data_type)
        else:
            fig = self.analyzer.plot_multiple_neurons_activity(self.analyzer.neuron_queue, data_type)
        
        if fig:
            self.activity_canvas = FigureCanvasTkAgg(fig, master=self.activity_frame)
            self.activity_canvas.draw()
            self.activity_toolbar = NavigationToolbar2Tk(self.activity_canvas, self.activity_frame)
            self.activity_toolbar.update()
            self.activity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.status_var.set(f"Displaying {data_type} for {len(self.analyzer.neuron_queue)} neurons")
    
    def plot_running(self):
        if not self.analyzer.loaded:
            self.status_var.set("Error: Load data first")
            return
        
        if self.running_canvas:
            self.running_canvas.get_tk_widget().destroy()
            if self.running_toolbar:
                self.running_toolbar.destroy()
        
        fig = self.analyzer.plot_running_activity()
        
        if fig:
            self.running_canvas = FigureCanvasTkAgg(fig, master=self.running_frame)
            self.running_canvas.draw()
            self.running_toolbar = NavigationToolbar2Tk(self.running_canvas, self.running_frame)
            self.running_toolbar.update()
            self.running_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_map_click(self, event):
        if not self.analyzer.loaded or event.inaxes is None:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        if 0 <= x < self.analyzer.Lx and 0 <= y < self.analyzer.Ly:
            neuron_id = self.analyzer.combined_mask[y, x]
            if neuron_id > 0:
                neuron_idx = neuron_id - 1
                
                if event.button == 1:
                    self.analyzer.selected_neuron_idx = neuron_idx
                    self.neuron_var.set(str(neuron_idx))
                    
                    if neuron_idx not in self.analyzer.neuron_queue:
                        self.analyzer.neuron_queue.append(neuron_idx)
                        self.update_queue_listbox()
                    
                    self.status_var.set(f"Selected neuron {neuron_idx}")
                
                elif event.button == 3:
                    if neuron_idx in self.analyzer.neuron_queue:
                        self.analyzer.neuron_queue.remove(neuron_idx)
                        
                        if self.analyzer.selected_neuron_idx == neuron_idx:
                            if self.analyzer.neuron_queue:
                                self.analyzer.selected_neuron_idx = self.analyzer.neuron_queue[0]
                                self.neuron_var.set(str(self.analyzer.neuron_queue[0]))
                            else:
                                self.analyzer.selected_neuron_idx = -1
                                self.neuron_var.set('')
                        
                        self.status_var.set(f"Removed neuron {neuron_idx} from queue")
                    else:
                        self.analyzer.neuron_queue.append(neuron_idx)
                        self.status_var.set(f"Added neuron {neuron_idx} to queue")
                    
                    self.update_queue_listbox()
                
                self.plot_neuron_map()
                self.plot_activity()

    def plot_cross_relation_analysis(self):
        if not self.analyzer.loaded:
            self.status_var.set("Error: Load data first")
            return
        
        threshold = self.threshold_var.get()
        data_type = self.data_type_var.get()
        reference_channel = self.reference_var.get()
        
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title(f"Cross-relation Analysis ({reference_channel}, {data_type})")
        analysis_window.geometry("1500x800")
        
        fig = self.analyzer.plot_cross_relation_analysis(
            threshold=threshold, 
            data_type=data_type, 
            reference_channel=reference_channel
        )
        
        if fig is None:
            self.status_var.set(f"No {reference_channel} neurons for cross-relation analysis")
            analysis_window.destroy()
            return
        
        canvas = FigureCanvasTkAgg(fig, master=analysis_window)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, analysis_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set(f"Displaying cross-relation analysis (channel: {reference_channel}, data: {data_type}, threshold: {threshold})")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = CalciumImagingApp(root)
    root.mainloop()