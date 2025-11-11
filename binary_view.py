# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 19:54:13 2025

@author: Pulee
"""

"""
Visualise all frames in suite2p's data.bin
Author : You
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import ndimage
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Navigate to the suite2p plane folder
suite2p_dir = r"E:\Calcium\1366\20251104\b\output\ch1\suite2p\plane0"
os.chdir(suite2p_dir)

# Read metadata from ops.npy
ops = np.load('ops.npy', allow_pickle=True).item()
Ly, Lx = ops['Ly'], ops['Lx']
dtype = np.dtype(ops.get('data_type', 'uint16'))  # fallback for older suites

# Load raw binary and reshape to (n_frames, Ly, Lx)
raw = np.fromfile('data.bin', dtype=dtype)
n_frames = raw.size // (Ly * Lx)
movie = raw.reshape((n_frames, Ly, Lx))
print(f"Movie shape: {movie.shape} | range: {movie.min()} -> {movie.max()}")

def compute_mean_image(movie):
    mean_img = movie.mean(axis=0).astype(np.float32)
    p5, p95 = np.percentile(mean_img, [5, 95])
    mean_img = (mean_img - p5) / (p95 - p5)
    mean_img = np.clip(mean_img, 0, 1)
    return mean_img

def compute_enhanced_mean_image(movie, cell_diameter=10):
    mean_img = movie.mean(axis=0).astype(np.float32)
    
    sigma_background = cell_diameter * 2
    low_freq = ndimage.gaussian_filter(mean_img, sigma=sigma_background)
    
    enhanced = mean_img - low_freq
    
    p1, p99 = np.percentile(enhanced, [1, 99])
    enhanced = (enhanced - p1) / (p99 - p1)
    enhanced = np.clip(enhanced, 0, 1)
    
    gamma = 1.2
    enhanced = np.power(enhanced, gamma)
    
    return enhanced

def compute_correlation_map(movie, subsample=10):
    if subsample > 1:
        movie_subsampled = movie[::subsample]
    else:
        movie_subsampled = movie
    
    movie_flat = movie_subsampled.reshape(movie_subsampled.shape[0], -1)
    
    max_pixels = min(10000, movie_flat.shape[1])
    indices = np.random.choice(movie_flat.shape[1], max_pixels, replace=False)
    movie_selected = movie_flat[:, indices]
    
    corr_matrix = np.corrcoef(movie_selected.T)
    
    correlation_map = np.zeros(Ly * Lx)
    for i, idx in enumerate(indices):
        corr_values = corr_matrix[i]
        corr_values[i] = 0
        correlation_map[idx] = np.mean(corr_values[corr_values != 0])
    
    correlation_map = correlation_map.reshape(Ly, Lx)
    
    correlation_map = np.nan_to_num(correlation_map)
    
    if correlation_map.max() > correlation_map.min():
        correlation_map = (correlation_map - correlation_map.min()) / (correlation_map.max() - correlation_map.min())
    
    return correlation_map

def compute_max_projection(movie):
    max_img = movie.max(axis=0).astype(np.float32)
    p1, p99 = np.percentile(max_img, [1, 99])
    max_img = (max_img - p1) / (p99 - p1)
    max_img = np.clip(max_img, 0, 1)
    return max_img

print("Computing mean image...")
mean_img = compute_mean_image(movie)

print("Computing enhanced mean image...")
enhanced_mean_img = compute_enhanced_mean_image(movie)

print("Computing correlation map...")
correlation_map = compute_correlation_map(movie)

print("Computing max projection...")
max_proj = compute_max_projection(movie)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Suite2P-like Image Processing', fontsize=16, fontweight='bold')

fig.delaxes(axes[1, 2])

ax1, ax2, ax3 = axes[0, :]
ax4, ax5 = axes[1, :2]

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.axis('off')

p1, p99 = np.percentile(movie, [1, 99])
img1 = ax1.imshow(movie[0], cmap='gray', vmin=p1, vmax=p99)
ax1.set_title('Raw Frame (Animation)')

img2 = ax2.imshow(mean_img, cmap='gray')
ax2.set_title('Mean Image')

img3 = ax3.imshow(enhanced_mean_img, cmap='gray')
ax3.set_title('Enhanced Mean Image')

img4 = ax4.imshow(correlation_map, cmap='viridis')
ax4.set_title('Correlation Map')

img5 = ax5.imshow(max_proj, cmap='gray')
ax5.set_title('Max Projection')

plt.colorbar(img4, ax=ax4, fraction=0.046, pad=0.04)
plt.colorbar(img5, ax=ax5, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.subplots_adjust(top=0.9)

def update(idx):
    img1.set_data(movie[idx])
    ax1.set_title(f'Raw Frame {idx+1:04d}/{n_frames}')
    return img1,

ani = FuncAnimation(fig, update, frames=min(n_frames, 1000), interval=50, blit=True)

plt.show()

save_images = False
if save_images:
    plt.imsave('mean_image.png', mean_img, cmap='gray')
    plt.imsave('enhanced_mean_image.png', enhanced_mean_img, cmap='gray')
    plt.imsave('correlation_map.png', correlation_map, cmap='viridis')
    plt.imsave('max_projection.png', max_proj, cmap='gray')
    print("Images saved!")