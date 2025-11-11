import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os

base_dir = r"F:\calcium data\animal1\20250804\aoutput"
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

print(f"Total neurons: {len(combined_neurons)}")
print(f"Channel 1 only: {len(neurons_ch1) - len(matched_ch1)}")
print(f"Channel 2 only: {len(neurons_ch2) - len(matched_ch2)}")
print(f"overlap: {len(overlap_pairs)}")

def combine_activity_data():
    combined_activity = []
    
    for neuron in data_split_neu_type:
        neuron_data = {
            'type': neuron['type'],
            'mask': neuron['mask'],
            'coords': neuron['coords'],
            'color': neuron.get('color', 'gray')
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