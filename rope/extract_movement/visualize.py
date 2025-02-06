import json
import numpy as np
import torch
from matplotlib import pyplot as plt

def visualize_self_attention(matrix, fix_bar=True, directory='rope', axis = 'xy'):

    frames,widths,heights = matrix.shape
    
    max_value = np.max(matrix).item()
    
    col = 4
    row = frames // col + 1
    fig, axs = plt.subplots(row, col, figsize=(col*5,row*5))
    
    for j in range(frames):
        ax_a = axs[j // col, j % col]
        
        
        current_map = matrix[j]
        
        im_a = ax_a.imshow(current_map, cmap='jet', aspect='auto')
        cbar = fig.colorbar(im_a, ax=ax_a)
        if fix_bar:
            im_a.set_clim(0, max_value) 
        
            
        ax_a.axis('off')  # Hide the axes
        ax_a.set_title(f'Frame {j+1}')
        
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    path = directory + f'_.png'
    fig.savefig(path)
    
path = 'rope/extract_movement/trace_matrix.json'
matrix = json.load(open(path))
matrix = np.array(matrix)

# matrix[:,60:68,32:40] = 0
visualize_self_attention(matrix, fix_bar=True, directory='rope/extract_movement/matrix_')

hisgram = np.zeros((25,36))
placeholder = np.zeros((128,72))
for i in range(128):
    for j in range(72):
        placeholder[i,j] = (i-64)**2 + (j-36)**2

for i in range(25):
    for j in range(36):
        hisgram[i,j] = np.sum(matrix[i][(placeholder >= j**2) & (placeholder < (j+1)**2)])

fig, axs = plt.subplots(5,5, figsize=(25,25))
axs = axs.flatten()
# compute the p percentile of the hisgram
p = 0.9
percentiles = np.zeros(25)
hisgram_copy = hisgram.copy()
# hisgram_copy[:,0:3]=0
for j in range(25):
    total_sum = np.sum(hisgram_copy[j])
    target_value = p * total_sum
    min_diff = 1e10
    cumulative_sum = 0
    k = -1
    for i in range (36):
        cumulative_sum = cumulative_sum + hisgram_copy[j][i]
        diff = abs(cumulative_sum - target_value)
        
        if diff < min_diff:
            min_diff = diff
            k = i
    percentiles[j] = k
print(percentiles)
print([int(i/25*9) for i in percentiles])

max_value = np.max(hisgram)
for i in range(25):
    # create a bar chart
    # print(np.sum(hisgram[i]))
    axs[i].bar(range(36), hisgram[i])
    axs[i].set_title(f'Frame {i+1}')
    # fix the y-axis to be max_value
    axs[i].set_ylim(0, max_value)
    # add a line to show the p percentile
    axs[i].axvline(x=percentiles[i], color='r')
fig.tight_layout()
fig.savefig('rope/extract_movement/hisgram.png')
