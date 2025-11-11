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
from matplotlib.widgets import RectangleSelector

class CalciumImagingAnalyzer:
    def __init__(self):
        self.base_dir = ""
        self.running_path = ""
        self.sync_folder = ""
        self.experiment_xml_file = ""
        self.OVERLAP_THRESHOLD = 0.5
        self.loaded = False
        self.selected_neuron_idx = -1
        self.selected_neuron_indices = []  # For multiple selection
        
    def auto_detect_files(self, base_directory):
        result = {
            'base_dir': base_directory,
            'ch1_path': None,
            'ch2_path': None,
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
            ch1_path = os.path.join(output_dir, 'ch1', 'suite2p', 'plane0')
            ch2_path = os.path.join(output_dir, 'ch2', 'suite2p', 'plane0')
            
            if os.path.exists(ch1_path):
                result['ch1_path'] = ch1_path
            if os.path.exists(ch2_path):
                result['ch2_path'] = ch2_path
        
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
            if not file_paths['ch1_path']:
                missing_files.append('ch1 data')
            if not file_paths['ch2_path']:
                missing_files.append('ch2 data')
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
            
            ch1_path = file_paths['ch1_path']
            ch2_path = file_paths['ch2_path']
            
            self.stat_chan1 = np.load(os.path.join(ch1_path, "stat.npy"), allow_pickle=True)
            self.ops_chan1 = np.load(os.path.join(ch1_path, "ops.npy"), allow_pickle=True).item()
            self.F_chan1 = np.load(os.path.join(ch1_path, "F.npy"))
            self.Fneu_chan1 = np.load(os.path.join(ch1_path, "Fneu.npy"))
            self.spks_chan1 = np.load(os.path.join(ch1_path, "spks.npy"))
            self.iscell_chan1 = np.load(os.path.join(ch1_path, "iscell.npy"))
            
            self.stat_chan2 = np.load(os.path.join(ch2_path, "stat.npy"), allow_pickle=True)
            self.ops_chan2 = np.load(os.path.join(ch2_path, "ops.npy"), allow_pickle=True).item()
            self.F_chan2 = np.load(os.path.join(ch2_path, "F.npy"))
            self.Fneu_chan2 = np.load(os.path.join(ch2_path, "Fneu.npy"))
            self.spks_chan2 = np.load(os.path.join(ch2_path, "spks.npy"))
            self.iscell_chan2 = np.load(os.path.join(ch2_path, "iscell.npy"))
            
            self.Ly = self.ops_chan1['Ly']
            self.Lx = self.ops_chan1['Lx']
            assert self.Ly == self.ops_chan2['Ly'] and self.Lx == self.ops_chan2['Lx'], "Channel dimensions don't match!"
            
            data = np.fromfile(self.running_path, dtype=np.float64)
            data = data.reshape(-1, 100, 5)
            self.running_data = data[:, :, 1].ravel() * 7.5 / 360
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
            
            self.sample_rate = 10000.0  
            
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
            
            self.neurons_ch1 = self.preprocess_neurons(self.stat_chan1, self.iscell_chan1)
            self.neurons_ch2 = self.preprocess_neurons(self.stat_chan2, self.iscell_chan2)
            
            self.match_neurons()
            
            self.create_combined_mask()
            
            self.combined_activity = self.combine_activity_data()
            
            n_frames = self.F_chan1.shape[1]
            self.imaging_times = np.arange(n_frames) / self.frame_rate * self.average_num
            
            valid_mask = (self.running_time >= 0) & (self.running_time <= self.imaging_times[-1])
            self.running_time_valid = self.running_time[valid_mask]
            self.running_data_valid = self.running_data[valid_mask]
            
            self.loaded = True
            print("Data loaded successfully!")
            print(f"Total neurons: {len(self.combined_neurons)}")
            print(f"Channel 1 only: {len(self.neurons_ch1) - len(self.matched_ch1)}")
            print(f"Channel 2 only: {len(self.neurons_ch2) - len(self.matched_ch2)}")
            print(f"Overlap: {len(self.overlap_pairs)}")
            
            return True, "Data loaded successfully!"
            
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def preprocess_neurons(self, stat, iscell):
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
                    'index': i
                })
        return neurons
    
    def match_neurons(self):
        self.matched_ch1 = set()
        self.matched_ch2 = set()
        self.overlap_pairs = []
        
        for i, n1 in enumerate(self.neurons_ch1):
            for j, n2 in enumerate(self.neurons_ch2):
                overlap = np.sum(n1['mask'] & n2['mask'])
                if overlap > 0:
                    min_area = min(n1['area'], n2['area'])
                    overlap_ratio = overlap / min_area
                    if overlap_ratio >= self.OVERLAP_THRESHOLD:
                        self.overlap_pairs.append((i, j, overlap_ratio))
                        self.matched_ch1.add(i)
                        self.matched_ch2.add(j)
    
    def create_combined_mask(self):
        self.combined_neurons = []
        
        # Channel 1 only neurons
        for i, n in enumerate(self.neurons_ch1):
            if i not in self.matched_ch1:
                neuron_data = {
                    'mask': n['mask'],
                    'coords': n['coords'],
                    'type': 'ch1_only',
                    'color': 'green',
                    'orig_index': n['index'],
                    'orig_channel': 'ch1'
                }
                self.combined_neurons.append(neuron_data)
        
        # Channel 2 only neurons
        for j, n in enumerate(self.neurons_ch2):
            if j not in self.matched_ch2:
                neuron_data = {
                    'mask': n['mask'],
                    'coords': n['coords'],
                    'type': 'ch2_only',
                    'color': 'red',
                    'orig_index': n['index'],
                    'orig_channel': 'ch2'
                }
                self.combined_neurons.append(neuron_data)
        
        # Overlap neurons
        for i, j, ratio in self.overlap_pairs:
            n1 = self.neurons_ch1[i]
            n2 = self.neurons_ch2[j]
            
            if n1['area'] >= n2['area']:
                selected = n1
                channel = 'ch1'
                orig_index = n1['index']
            else:
                selected = n2
                channel = 'ch2'
                orig_index = n2['index']
            
            neuron_data = {
                'mask': selected['mask'],
                'coords': selected['coords'],
                'type': 'overlap',
                'color': 'yellow',
                'orig_index': orig_index,
                'orig_channel': channel,
                'overlap_ratio': ratio,
                'orig_index_ch1': n1['index'],
                'orig_index_ch2': n2['index']
            }
            self.combined_neurons.append(neuron_data)
        
        # Create combined mask image
        self.combined_mask = np.zeros((self.Ly, self.Lx), dtype=int)
        for idx, neuron in enumerate(self.combined_neurons):
            ypix, xpix = neuron['coords']
            self.combined_mask[ypix, xpix] = idx + 1
        
        # Create colormap
        colors = ['black']  # background
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
            
            if neuron['type'] == 'ch1_only':
                idx = neuron['orig_index']
                neuron_data['F_ch1'] = self.F_chan1[idx]
                neuron_data['Fneu_ch1'] = self.Fneu_chan1[idx]
                neuron_data['spks_ch1'] = self.spks_chan1[idx]
                neuron_data['F_ch2'] = None
                neuron_data['spks_ch2'] = None
                
            elif neuron['type'] == 'ch2_only':
                idx = neuron['orig_index']
                neuron_data['F_ch2'] = self.F_chan2[idx]
                neuron_data['Fneu_ch2'] = self.Fneu_chan2[idx]
                neuron_data['spks_ch2'] = self.spks_chan2[idx]
                neuron_data['F_ch1'] = None
                neuron_data['spks_ch1'] = None
                
            elif neuron['type'] == 'overlap':
                idx_ch1 = neuron['orig_index_ch1']
                idx_ch2 = neuron['orig_index_ch2']
                
                neuron_data['F_ch1'] = self.F_chan1[idx_ch1]
                neuron_data['Fneu_ch1'] = self.Fneu_chan1[idx_ch1]
                neuron_data['spks_ch1'] = self.spks_chan1[idx_ch1]
                neuron_data['F_ch2'] = self.F_chan2[idx_ch2]
                neuron_data['Fneu_ch2'] = self.Fneu_chan2[idx_ch2]
                neuron_data['spks_ch2'] = self.spks_chan2[idx_ch2]
            
            if 'overlap_ratio' in neuron:
                neuron_data['overlap_ratio'] = neuron['overlap_ratio']
            
            combined_activity.append(neuron_data)
        
        return combined_activity
    
    def plot_neuron_activity(self, neuron_indices):
        if not self.loaded or not neuron_indices:
            return None
        
        fig = Figure(figsize=(10, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(neuron_indices)))
        
        for i, neuron_idx in enumerate(neuron_indices):
            if neuron_idx < 0 or neuron_idx >= len(self.combined_activity):
                continue
                
            neuron = self.combined_activity[neuron_idx]
            neuron_type = neuron['type']
            
            # Plot calcium activity
            if neuron_type == 'ch1_only':
                ax.plot(self.imaging_times, neuron['Fneu_ch1'], '-', color=colors[i], 
                        label=f'Neuron {neuron_idx} (Ch1)', alpha=0.8)
            elif neuron_type == 'ch2_only':
                ax.plot(self.imaging_times, neuron['Fneu_ch2'], '-', color=colors[i], 
                        label=f'Neuron {neuron_idx} (Ch2)', alpha=0.8)
            else:  # overlap
                ax.plot(self.imaging_times, neuron['Fneu_ch1'], '-', color=colors[i], 
                        label=f'Neuron {neuron_idx} (Overlap)', alpha=0.8)
        
        ax.set_ylabel('Fluorescence neuropil (Fneu)')
        ax.set_title(f'Neuron Activity ({len(neuron_indices)} neurons)', fontsize=14)
        ax.grid(False)
        ax.legend(loc='upper right')
        
        # Set x-axis limits
        ax.set_xlim(self.imaging_times[0], self.imaging_times[-1])
        
        fig.tight_layout()
        return fig
    
    def plot_running_signal(self):
        if not self.loaded:
            return None
        
        fig = Figure(figsize=(10, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(self.running_time_valid, self.running_data_valid, 'b-', alpha=0.8, label='Running Speed')
        ax.set_ylabel('Running Speed (cm/s)', color='b')
        ax.set_xlabel('Time (s)')
        ax.set_title('Running Activity', fontsize=14)
        ax.grid(False)
        ax.legend(loc='upper right')
        
        # Set x-axis limits to match activity plot
        ax.set_xlim(self.imaging_times[0], self.imaging_times[-1])
        
        fig.tight_layout()
        return fig
    
    def plot_neuron_mask(self, highlight_indices=[]):
        if not self.loaded:
            return None
        
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Display neuron mask
        img = ax.imshow(self.combined_mask, cmap=self.cmap, 
                        vmin=0, vmax=len(self.combined_neurons), 
                        interpolation='nearest')
        
        # Highlight selected neurons
        for neuron_idx in highlight_indices:
            if 0 <= neuron_idx < len(self.combined_neurons):
                neuron = self.combined_neurons[neuron_idx]
                ypix, xpix = neuron['coords']
                # Create a mask for the selected neuron
                highlight_mask = np.zeros_like(self.combined_mask)
                highlight_mask[ypix, xpix] = 1
                # Overlay in cyan with transparency
                ax.imshow(highlight_mask, cmap=ListedColormap(['none', 'cyan']), 
                         alpha=0.5, interpolation='nearest')
        
        # Create legend
        legend_patches = [
            mpatches.Patch(color='green', label=f'Ch1 Only ({len(self.neurons_ch1) - len(self.matched_ch1)})'),
            mpatches.Patch(color='red', label=f'Ch2 Only ({len(self.neurons_ch2) - len(self.matched_ch2)})'),
            mpatches.Patch(color='yellow', label=f'Overlap ({len(self.overlap_pairs)})')
        ]
        ax.legend(handles=legend_patches, loc='upper right')
        
        ax.set_title(f'Dual-Channel Neuron Map ({len(self.combined_neurons)} neurons)', fontsize=14)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        fig.tight_layout()
        return fig


class CalciumImagingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual-Channel Calcium Imaging Analyzer")
        self.root.geometry("1500x900")
        self.analyzer = CalciumImagingAnalyzer()
        
        self.left_canvas = None
        self.left_toolbar = None
        self.top_right_canvas = None
        self.top_right_toolbar = None
        self.bottom_right_canvas = None
        self.bottom_right_toolbar = None
        
        self.rect_selector = None
        self.selection_mode = "single"  # "single" or "rectangle"
        
        self.create_widgets()
    
    def create_widgets(self):
        # Create main frames
        control_frame = tk.Frame(self.root, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        display_frame = tk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left and right display frames
        left_display_frame = tk.Frame(display_frame, width=600)
        left_display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_display_frame = tk.Frame(display_frame, width=600)
        right_display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create top and bottom right display frames
        top_right_display_frame = tk.Frame(right_display_frame, height=400)
        top_right_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        bottom_right_display_frame = tk.Frame(right_display_frame, height=300)
        bottom_right_display_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Store frame references
        self.left_display_frame = left_display_frame
        self.top_right_display_frame = top_right_display_frame
        self.bottom_right_display_frame = bottom_right_display_frame
        
        # Control panel
        tk.Label(control_frame, text="Data Loading", font=("Arial", 12, "bold")).pack(pady=(10, 5), anchor="w")
        
        # Directory selection button
        dir_frame = tk.Frame(control_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(dir_frame, text="Select Experiment Directory", 
                 command=self.select_experiment_dir).pack(fill=tk.X, pady=2)
        
        # Load button
        tk.Button(control_frame, text="Load Data", command=self.load_data,
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(pady=10, fill=tk.X)
        
        # Selection tools
        tk.Label(control_frame, text="Selection Tools", font=("Arial", 12, "bold")).pack(pady=(20, 5), anchor="w")
        
        # Single selection button
        tk.Button(control_frame, text="Single Selection Mode", 
                 command=lambda: self.set_selection_mode("single"), bg="#4CAF50", fg="white").pack(fill=tk.X, pady=2)
        
        # Rectangle selection button
        tk.Button(control_frame, text="Rectangle Selection Mode", 
                 command=lambda: self.set_selection_mode("rectangle"), bg="#2196F3", fg="white").pack(fill=tk.X, pady=2)
        
        # Clear selection button
        tk.Button(control_frame, text="Clear Selection", 
                 command=self.clear_selection, bg="#FF9800", fg="white").pack(fill=tk.X, pady=2)
        
        # Selected neurons list
        tk.Label(control_frame, text="Selected Neurons", font=("Arial", 12, "bold")).pack(pady=(20, 5), anchor="w")
        
        self.selected_listbox = tk.Listbox(control_frame, height=10)
        self.selected_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.selected_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Single Selection Mode")
        tk.Label(control_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def set_selection_mode(self, mode):
        self.selection_mode = mode
        if mode == "single":
            self.status_var.set("Single Selection Mode - Click on neurons to select")
            if self.rect_selector:
                self.rect_selector.set_active(False)
        else:
            self.status_var.set("Rectangle Selection Mode - Drag to select multiple neurons")
            if self.rect_selector:
                self.rect_selector.set_active(True)
        
        # Redraw the neuron map to update the selection mode
        if self.analyzer.loaded:
            self.plot_neuron_map()
    
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
                
                # Plot neuron map and activity
                self.plot_neuron_map()
                self.plot_activity()
                self.plot_running()
            else:
                self.status_var.set(f"Error: {message}")
                messagebox.showerror("Error", message)
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def on_listbox_select(self, event):
        # Get selected indices from listbox
        selected_indices = self.selected_listbox.curselection()
        if selected_indices:
            # Convert to neuron indices (stored as strings in listbox)
            self.analyzer.selected_neuron_indices = [int(self.selected_listbox.get(i)) for i in selected_indices]
            self.plot_neuron_map()
            self.plot_activity()
    
    def on_rectangle_select(self, eclick, erelease):
        if not self.analyzer.loaded:
            return
        
        # Get rectangle coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, x2, self.analyzer.Lx-1))
        x2 = min(self.analyzer.Lx-1, max(x1, x2))
        y1 = max(0, min(y1, y2, self.analyzer.Ly-1))
        y2 = min(self.analyzer.Ly-1, max(y1, y2))
        
        # Convert to integer coordinates
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)
        
        # Find neurons in the selected rectangle
        selected_neurons = set()
        for y in range(y1, y2+1):
            for x in range(x1, x2+1):
                neuron_id = self.analyzer.combined_mask[y, x]
                if neuron_id > 0:
                    selected_neurons.add(neuron_id - 1)  # Convert from mask index to neuron index
        
        # Update selection
        self.analyzer.selected_neuron_indices = list(selected_neurons)
        
        # Update listbox
        self.update_selection_listbox()
        
        # Update plots
        self.plot_neuron_map()
        self.plot_activity()
        
        self.status_var.set(f"Selected {len(self.analyzer.selected_neuron_indices)} neurons")
    
    def clear_selection(self):
        self.analyzer.selected_neuron_indices = []
        self.update_selection_listbox()
        self.plot_neuron_map()
        self.plot_activity()
        self.status_var.set("Selection cleared")
    
    def update_selection_listbox(self):
        self.selected_listbox.delete(0, tk.END)
        for neuron_idx in sorted(self.analyzer.selected_neuron_indices):
            self.selected_listbox.insert(tk.END, str(neuron_idx))
    
    def plot_neuron_map(self):
        if not self.analyzer.loaded:
            self.status_var.set("Error: Load data first")
            return
        
        # Clear previous left display
        if self.left_canvas:
            self.left_canvas.get_tk_widget().destroy()
            if self.left_toolbar:
                self.left_toolbar.destroy()
        
        fig = self.analyzer.plot_neuron_mask(self.analyzer.selected_neuron_indices)
        
        # Create new canvas and toolbar
        self.left_canvas = FigureCanvasTkAgg(fig, master=self.left_display_frame)
        self.left_canvas.draw()
        
        self.left_toolbar = NavigationToolbar2Tk(self.left_canvas, self.left_display_frame)
        self.left_toolbar.update()
        self.left_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add click event handler for single selection
        self.left_canvas.mpl_connect('button_press_event', self.on_map_click)
        
        # Add rectangle selector if in rectangle mode
        if self.selection_mode == "rectangle":
            ax = fig.gca()
            try:
                self.rect_selector = RectangleSelector(
                    ax, self.on_rectangle_select,
                    useblit=True,
                    button=[1],  # Left mouse button
                    minspanx=5, minspany=5,  # Minimum span in pixels
                    spancoords='pixels',
                    interactive=True,
                    rectprops=dict(facecolor='red', edgecolor='black', alpha=0.2, fill=True)
                )
            except TypeError:
                self.rect_selector = RectangleSelector(
                    ax, self.on_rectangle_select,
                    useblit=True,
                    button=[1],
                    minspanx=5, minspany=5,
                    spancoords='pixels',
                    interactive=True
                )
                try:
                    self.rect_selector.rect.set_facecolor('red')
                    self.rect_selector.rect.set_alpha(0.2)
                    self.rect_selector.rect.set_edgecolor('black')
                except AttributeError:
                    try:
                        self.rect_selector.current_rect.set_facecolor('red')
                        self.rect_selector.current_rect.set_alpha(0.2)
                        self.rect_selector.current_rect.set_edgecolor('black')
                    except AttributeError:
                        print("Warning: Cannot set rectangle properties in this matplotlib version")
            
            self.status_var.set("Rectangle selection enabled. Drag to select neurons.")
        else:
            if self.rect_selector:
                self.rect_selector.set_active(False)
            self.status_var.set("Single selection mode. Click on neurons to select.")
        
        self.status_var.set(f"Displaying neuron map. Total neurons: {len(self.analyzer.combined_neurons)}")
    
    def plot_activity(self):
        if not self.analyzer.loaded:
            self.status_var.set("Error: Load data first")
            return
        
        if not self.analyzer.selected_neuron_indices:
            # Clear activity plot if no neurons selected
            if self.top_right_canvas:
                self.top_right_canvas.get_tk_widget().destroy()
                if self.top_right_toolbar:
                    self.top_right_toolbar.destroy()
            return
        
        # Clear previous top right display
        if self.top_right_canvas:
            self.top_right_canvas.get_tk_widget().destroy()
            if self.top_right_toolbar:
                self.top_right_toolbar.destroy()
        
        fig = self.analyzer.plot_neuron_activity(self.analyzer.selected_neuron_indices)
        
        # Create new canvas and toolbar
        self.top_right_canvas = FigureCanvasTkAgg(fig, master=self.top_right_display_frame)
        self.top_right_canvas.draw()
        
        self.top_right_toolbar = NavigationToolbar2Tk(self.top_right_canvas, self.top_right_display_frame)
        self.top_right_toolbar.update()
        self.top_right_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set(f"Displaying activity for {len(self.analyzer.selected_neuron_indices)} neurons")
    
    def plot_running(self):
        if not self.analyzer.loaded:
            self.status_var.set("Error: Load data first")
            return
        
        # Clear previous bottom right display
        if self.bottom_right_canvas:
            self.bottom_right_canvas.get_tk_widget().destroy()
            if self.bottom_right_toolbar:
                self.bottom_right_toolbar.destroy()
        
        fig = self.analyzer.plot_running_signal()
        
        # Create new canvas and toolbar
        self.bottom_right_canvas = FigureCanvasTkAgg(fig, master=self.bottom_right_display_frame)
        self.bottom_right_canvas.draw()
        
        self.bottom_right_toolbar = NavigationToolbar2Tk(self.bottom_right_canvas, self.bottom_right_display_frame)
        self.bottom_right_toolbar.update()
        self.bottom_right_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var.set("Displaying running signal")
    
    def on_map_click(self, event):
        if not self.analyzer.loaded or event.inaxes is None or self.selection_mode != "single":
            return
            
        # Get clicked coordinates
        x, y = int(event.xdata), int(event.ydata)
        
        # Find neuron at clicked position
        if 0 <= x < self.analyzer.Lx and 0 <= y < self.analyzer.Ly:
            neuron_id = self.analyzer.combined_mask[y, x]
            if neuron_id > 0:
                neuron_idx = neuron_id - 1  # Convert from mask index to neuron index
                self.analyzer.selected_neuron_indices = [neuron_idx]
                self.update_selection_listbox()
                self.status_var.set(f"Selected neuron {neuron_idx}")
                
                # Update displays
                self.plot_neuron_map()
                self.plot_activity()


if __name__ == "__main__":
    root = tk.Tk()
    app = CalciumImagingApp(root)
    root.mainloop()