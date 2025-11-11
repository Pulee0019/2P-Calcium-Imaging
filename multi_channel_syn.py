import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os
import h5py
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ET

base_dir = r"F:\PCW's program\calcium imaging\calcium data\20250630\a\output"
ch1_path = os.path.join(base_dir, "ch1", "suite2p", "plane0")
ch2_path = os.path.join(base_dir, "ch2", "suite2p", "plane0")

stat_chan1 = np.load(os.path.join(ch1_path, "stat.npy"), allow_pickle=True)
ops_chan1 = np.load(os.path.join(ch1_path, "ops.npy"), allow_pickle=True).item()
F_chan1 = np.load(os.path.join(ch1_path, "F.npy"))
Fneu_chan1 = np.load(os.path.join(ch1_path, "Fneu.npy"))
spks_chan1 = np.load(os.path.join(ch1_path, "spks.npy"))
iscell_chan1 = np.load(os.path.join(ch1_path, "iscell.npy"))

stat_chan2 = np.load(os.path.join(ch2_path, "stat.npy"), allow_pickle=True)
ops_chan2 = np.load(os.path.join(ch2_path, "ops.npy"), allow_pickle=True).item()
F_chan2 = np.load(os.path.join(ch2_path, "F.npy"))
Fneu_chan2 = np.load(os.path.join(ch2_path, "Fneu.npy"))
spks_chan2 = np.load(os.path.join(ch2_path, "spks.npy"))
iscell_chan2 = np.load(os.path.join(ch2_path, "iscell.npy"))

Ly = ops_chan1['Ly']
Lx = ops_chan1['Lx']
assert Ly == ops_chan2['Ly'] and Lx == ops_chan2['Lx'], "Channel dimensions don't match!"

running_path = r"F:\PCW's program\calcium imaging\calcium data\20250630\a\filename_running_1.dat2"
sync_folder = r"F:\PCW's program\calcium imaging\calcium data\20250630\a\SyncData_0007"
sync_file = os.path.join(sync_folder, "Episode_0000.h5")

data = np.fromfile(running_path, dtype=np.float64)
data = data.reshape(-1, 100, 5)
running_data = data[:, :, 1].ravel()
raw_running_time = data[:, :, 0].ravel()
running_relative_time = (raw_running_time - raw_running_time[0])*24*60

with h5py.File(sync_file, 'r') as h5f:
    running_sync = np.array(h5f['/AI/Runningdata'])
    frame_out = np.array(h5f['/DI/FrameOut'])
    trigger_signal = np.array(h5f['/DI/Triggersignal'])

real_time_xml_file = os.path.join(sync_folder, "ThorRealTimeDataSettings.xml")
tree = ET.parse(real_time_xml_file)
root = tree.getroot()

sample_rate = 10000.0
# for daq in root.findall('.//DaqDevices/AcquireBoard'):
#     for sample_rate_elem in daq.findall('SampleRate'):
#         if 'rate' in sample_rate_elem.attrib:
#             try:
#                 if int(sample_rate_elem.attrib['enable']):
#                     print(1)
#                     sample_rate = float(sample_rate_elem.attrib['rate'])
#                     print(sample_rate)
#                     break
#             except ValueError:
#                 continue
            
experiment_xml_file = r"F:\PCW's program\calcium imaging\calcium data\20250630\a\day4\Experiment.xml"
tree = ET.parse(experiment_xml_file)
root = tree.getroot()    
lsm = root.find('.//LSM[@name="ResonanceGalvo"]')

if lsm is not None:
    frame_rate = lsm.get('frameRate')
    frame_rate = float(frame_rate)
    average_num = lsm.get('averageNum')
    average_num = int(average_num)
else:
    print("LSM cannot find!")

running_onset_idx = np.where(trigger_signal > 0.5)[0]
imaging_onset_idx = np.where(frame_out > 0.5)[0]

if len(running_onset_idx) > 0 and len(imaging_onset_idx) > 0:
    running_onset = running_onset_idx[0]
    diff = np.diff(imaging_onset_idx)
    breaks = np.where(diff > 100)[0]
    starts = np.concatenate(([0], breaks + 1))
    first_indices = imaging_onset_idx[starts]
    imaging_onset = first_indices[-1]
    relative_time = (imaging_onset - running_onset) / sample_rate
else:
    relative_time = 0.0
    print("Warning: Trigger signals not found, using zero offset")

running_time = running_relative_time*60 - relative_time

OVERLAP_THRESHOLD = 0.5

def preprocess_neurons(stat, iscell):
    neurons = []
    for i, cell in enumerate(stat):
        if iscell[i, 1] > 0.0:
            mask = np.zeros((Ly, Lx), dtype=bool)
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

neurons_ch1 = preprocess_neurons(stat_chan1, iscell_chan1)
neurons_ch2 = preprocess_neurons(stat_chan2, iscell_chan2)

matched_ch1 = set()
matched_ch2 = set()
overlap_pairs = []

for i, n1 in enumerate(neurons_ch1):
    for j, n2 in enumerate(neurons_ch2):
        overlap = np.sum(n1['mask'] & n2['mask'])
        if overlap > 0:
            min_area = min(n1['area'], n2['area'])
            overlap_ratio = overlap / min_area
            if overlap_ratio >= OVERLAP_THRESHOLD:
                overlap_pairs.append((i, j, overlap_ratio))
                matched_ch1.add(i)
                matched_ch2.add(j)

combined_neurons = []
data_split_neu_type = []

for i, n in enumerate(neurons_ch1):
    if i not in matched_ch1:
        neuron_data = {
            'mask': n['mask'],
            'coords': n['coords'],
            'type': 'ch1_only',
            'color': 'green',
            'orig_index': n['index'],
            'orig_channel': 'ch1'
        }
        combined_neurons.append(neuron_data)
        data_split_neu_type.append(neuron_data)

for j, n in enumerate(neurons_ch2):
    if j not in matched_ch2:
        neuron_data = {
            'mask': n['mask'],
            'coords': n['coords'],
            'type': 'ch2_only',
            'color': 'red',
            'orig_index': n['index'],
            'orig_channel': 'ch2'
        }
        combined_neurons.append(neuron_data)
        data_split_neu_type.append(neuron_data)

for i, j, ratio in overlap_pairs:
    n1 = neurons_ch1[i]
    n2 = neurons_ch2[j]
    
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
    combined_neurons.append(neuron_data)
    data_split_neu_type.append(neuron_data)

fs = ops_chan1['fs']
n_frames = F_chan1.shape[1]

imaging_times = np.arange(n_frames) / frame_rate * average_num

valid_mask = (running_time >= 0) & (running_time <= imaging_times[-1])

running_time_valid = running_time[valid_mask]
running_data_valid = running_data[valid_mask]

def combine_activity_data():
    combined_activity = []
    
    for neuron in data_split_neu_type:
        neuron_data = {
            'type': neuron['type'],
            'mask': neuron['mask'],
            'coords': neuron['coords'],
            'color': neuron.get('color', 'gray'),
            'aligned_running_time': running_time_valid,
            'aligned_running': running_data_valid
        }
        
        if neuron['type'] == 'ch1_only':
            idx = neuron['orig_index']
            neuron_data['F_ch1'] = F_chan1[idx]
            neuron_data['Fneu_ch1'] = Fneu_chan1[idx]
            neuron_data['spks_ch1'] = spks_chan1[idx]
            neuron_data['F_ch2'] = np.array([])
            neuron_data['Fneu_ch2'] = np.array([])
            neuron_data['spks_ch2'] = np.array([])
            
        elif neuron['type'] == 'ch2_only':
            idx = neuron['orig_index']
            neuron_data['F_ch2'] = F_chan2[idx]
            neuron_data['Fneu_ch2'] = Fneu_chan2[idx]
            neuron_data['spks_ch2'] = spks_chan2[idx]
            neuron_data['F_ch1'] = np.array([])
            neuron_data['Fneu_ch1'] = np.array([])
            neuron_data['spks_ch1'] = np.array([])
            
        elif neuron['type'] == 'overlap':
            idx_ch1 = neuron['orig_index_ch1']
            idx_ch2 = neuron['orig_index_ch2']
            
            neuron_data['F_ch1'] = F_chan1[idx_ch1]
            neuron_data['Fneu_ch1'] = Fneu_chan1[idx_ch1]
            neuron_data['spks_ch1'] = spks_chan1[idx_ch1]
            
            neuron_data['F_ch2'] = F_chan2[idx_ch2]
            neuron_data['Fneu_ch2'] = Fneu_chan2[idx_ch2]
            neuron_data['spks_ch2'] = spks_chan2[idx_ch2]
        
        if 'overlap_ratio' in neuron:
            neuron_data['overlap_ratio'] = neuron['overlap_ratio']
        
        combined_activity.append(neuron_data)
    
    return combined_activity

combined_activity = combine_activity_data()

combined_mask = np.zeros((Ly, Lx), dtype=int)
for idx, neuron in enumerate(combined_neurons):
    ypix, xpix = neuron['coords']
    combined_mask[ypix, xpix] = idx + 1

colors = ['black']
for neuron in combined_neurons:
    colors.append(neuron['color'])
cmap = ListedColormap(colors)

plt.figure(figsize=(14, 12))
plt.imshow(combined_mask, cmap=cmap, vmin=0, vmax=len(combined_neurons), interpolation='nearest')

legend_patches = [
    mpatches.Patch(color='green', label='Channel 1 Only'),
    mpatches.Patch(color='red', label='Channel 2 Only'),
    mpatches.Patch(color='yellow', label='Overlap')
]
plt.legend(handles=legend_patches, loc='upper right')

plt.title(f'Dual-Channel Cell Classification ({len(neurons_ch1)} ch1, {len(neurons_ch2)} ch2, {len(overlap_pairs)} overlap)', fontsize=16)
plt.xlabel('X Position', fontsize=12)
plt.ylabel('Y Position', fontsize=12)

plt.tight_layout()
plt.show()

first_neuron = combined_activity[300]
neuron_type = first_neuron['type']

fig, ax1 = plt.subplots(figsize=(15, 8))

time_axis = imaging_times

if neuron_type == 'ch1_only':
    ax1.plot(time_axis, first_neuron['F_ch1'], 'g-', label='Ch1 F signal', alpha=0.8)
elif neuron_type == 'ch2_only':
    ax1.plot(time_axis, first_neuron['F_ch2'], 'r-', label='Ch2 F signal', alpha=0.8)
else:
    ax1.plot(time_axis, first_neuron['F_ch1'], 'g-', label='Ch1 F signal', alpha=0.7)
    ax1.plot(time_axis, first_neuron['F_ch2'], 'r-', label='Ch2 F signal', alpha=0.7)

ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('F', color='tab:green', fontsize=12)
ax1.tick_params(axis='y', labelcolor='tab:green')
ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(running_time_valid, running_data_valid, 'b-', label='Running speed', alpha=0.6)
ax2.set_ylabel('Running Speed(rad/s)', color='tab:blue', fontsize=12)
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title(f'Aligned Activity: Neuron 1 ({neuron_type})', fontsize=16, pad=20)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

info_text = (f"Neuron type: {neuron_type}\n"
             f"Original channel: {first_neuron.get('orig_channel', 'N/A')}\n"
             f"Index: {first_neuron.get('orig_index', 'N/A')}")
if 'overlap_ratio' in first_neuron:
    info_text += f"\nOverlap ratio: {first_neuron['overlap_ratio']:.2f}"

plt.annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)

plt.tight_layout()
plt.show()

print(f"Total neurons: {len(combined_neurons)}")
print(f"Channel 1 only: {len(neurons_ch1) - len(matched_ch1)}")
print(f"Channel 2 only: {len(neurons_ch2) - len(matched_ch2)}")
print(f"Overlap: {len(overlap_pairs)}")