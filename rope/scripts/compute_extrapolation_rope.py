import sys
sys.path.append('/nfs/yban/large-scale-video-generation')

from models.vision.layers.positional_encoding.vision_llama_3d_rope_positional_encoder import VisionLLaMA3DRopePositionalEncoder, VisionLLaMA3DRopePositionalEncoder_Disable_Spatial, VisionLLaMA3DRopePositionalEncoder_Disable_Temporal, VisionLLaMA3DRopePositionalEncoder_Mono_ALL, VisionLLaMA3DRopePositionalEncoder_Mono_Spatial, VisionLLaMA3DRopePositionalEncoder_Prod
import torch
from matplotlib import pyplot as plt
import numpy as np

def plot_rope(frame, height, width, token_shape, attn, name, fix_bar=True):
    videos = np.zeros((token_shape[0], 3, token_shape[1]*4, token_shape[2]*4))

    frames, _, img_height, img_width = videos.shape
    # shape of self_attention_maps: [frames, width, height, frames, width, height]
    self_attention_maps = attn[frame, height, width, :, :, :] 
    current_video = self_attention_maps
    max_value = torch.max(current_video).item()

    # make videos[frame, :, width*4:width*4+4, height*4:height*4+4] red
    videos[:, 0,  height*4:height*4+4, width*4:width*4+4,] = 0
    videos[:, 1,  height*4:height*4+4, width*4:width*4+4,] = 1
    videos[:, 2,  height*4:height*4+4, width*4:width*4+4] = 0
    videos[frame, 0, height*4:height*4+4, width*4:width*4+4] = 1
    videos[frame, 1, height*4:height*4+4, width*4:width*4+4] = 0
    videos[frame, 2, height*4:height*4+4, width*4:width*4+4] = 0

    col = 4
    row = frames // col
    if row == 0:
        row = 1
        col = frames
    if img_height < img_width:
        fig, axs = plt.subplots(row, col*2, figsize=(col*5*2,row*3))
    else:
        fig, axs = plt.subplots(row, col*2, figsize=(col*5*2,row*5))
    axs = axs.flatten()

    for j in range(frames):
        ax_a = axs[j // col * col * 2 + j % col ]
        ax_v = axs[j // col * col * 2 + j % col + col]
        
        video = videos[j]
        video = np.clip((video * 127.5 + 128), 0, 255).astype('uint8')
        video = video/256
        im_v = ax_v.imshow(video.transpose(1,2,0))
        
        
        current_map = current_video[j]
        if False:
            current_map = current_map.unsqueeze(0)
            current_map = torch.nn.functional.interpolate(current_map.unsqueeze(0), size=(img_height, img_width), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()
        else:
            current_map = current_map.cpu().numpy()
        
        # current_map = (current_map - np.min(current_map)) / (np.max(current_map) - np.min(current_map))
        # cmap = plt.get_cmap('jet')
        # rgba_data = cmap(current_map)
        # alpha = 1 - current_map
        # rgba_data[..., -1] = alpha
        
        im_a = ax_a.imshow(current_map, cmap='jet', aspect='auto')
        cbar = fig.colorbar(im_a, ax=ax_a)
        if fix_bar:
            im_a.set_clim(0, max_value) 
        # fig.colorbar(im_v, ax=ax_v)
        
            
        ax_a.axis('off')  # Hide the axes
        ax_v.axis('off') # Hide the axes
        ax_v.set_title(f'Frame {j+1}')
        ax_a.set_title(f'Frame {j+1}')
        

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    path = 'visualization_rope' + f'_{frame}_{width}_{height}_{token_shape}.{name}.self.png'
    fig.savefig(path)

def get_attn(token_shape, theta = 4.7, grid = None):
    
    token_shape_prod = torch.prod(torch.tensor(token_shape)).item()

    rope_encoder = VisionLLaMA3DRopePositionalEncoder_Prod({"grid_scale":grid}, 'cuda')
    query = torch.ones(1, token_shape_prod, 16, 64, device='cuda')
    key = torch.ones(1, token_shape_prod, 16, 64, device='cuda')

    precompu = rope_encoder.precompute_freqs_cis_3d(64, token_shape[0], token_shape[1], token_shape[2], 'cuda', theta = theta)

    query, key = rope_encoder(query, key, *token_shape, precompu)

    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    attn = query @ key.transpose(-2, -1) # B x n_heads x T x T

    # average over the head dimension moayed
    attn = attn.softmax(dim=-1)
    attn = attn.mean(1) # B x T x T
    attn = attn.reshape(*token_shape, *token_shape)
    return attn
    
    
attn = get_attn([24,16,16], theta = 4.7, grid=[1.0,1.0,1.0])
ratio = attn[0, 0, 0, 0, 0, 4] / attn[0, 0, 0, 0, 0, 0]
ratio_t = attn[0, 0, 0, 4, 0, 0] / attn[0, 0, 0, 0, 0, 0]
plot_rope(0, 0, 0, [24,16,16], attn, 'test', fix_bar=True)

# attn_new = get_attn([24,24,24], theta = 7, grid=[1.0,1.0,1.0])
# ratio_new = attn_new[0, 0, 0, 0, 0, 6] / attn_new[0, 0, 0, 0, 0, 0]
# ratio_new_t = attn_new[0, 0, 0, 4, 0, 0] / attn_new[0, 0, 0, 0, 0, 0]
# plot_rope(0, 0, 0, [24,24,24], attn_new, 'test_', fix_bar=True)
# print(ratio_new)
# print(ratio_new_t)

# # important
# attn_new = get_attn([24,24,24], theta = 7, grid=[1.5,1.0,1.0])
# ratio_new = attn_new[0, 0, 0, 0, 0, 6] / attn_new[0, 0, 0, 0, 0, 0]
# ratio_new_t = attn_new[0, 0, 0, 4, 0, 0] / attn_new[0, 0, 0, 0, 0, 0]
# plot_rope(0, 0, 0, [24,24,24], attn_new, 'test_', fix_bar=True)
# print(ratio_new)
# print(ratio_new_t)

attn_new = get_attn([24,24,24], theta = 4.7, grid=[1.0,1.0,1.0])
ratio_new = attn_new[0, 0, 0, 0, 0, 6] / attn_new[0, 0, 0, 0, 0, 0]
ratio_new_t = attn_new[0, 0, 0, 4, 0, 0] / attn_new[0, 0, 0, 0, 0, 0]
plot_rope(0, 0, 0, [24,24,24], attn_new, 'test_', fix_bar=True)
print(ratio_new)
print(ratio_new_t)

attn_new = get_attn([24,24,24], theta = 4.7, grid=[1.0,0.67,0.67])
ratio_new = attn_new[0, 0, 0, 0, 0, 6] / attn_new[0, 0, 0, 0, 0, 0]
ratio_new_t = attn_new[0, 0, 0, 4, 0, 0] / attn_new[0, 0, 0, 0, 0, 0]
plot_rope(0, 0, 0, [24,24,24], attn_new, 'test__', fix_bar=True)
print(ratio_new)
print(ratio_new_t)

# attn_new = get_attn([24,24,24], theta = 4.7, grid=[1.0,1.5,1.5])
# ratio_new = attn_new[0, 0, 0, 0, 0, 6] / attn_new[0, 0, 0, 0, 0, 0]
# plot_rope(0, 0, 0, [24,24,24], attn_new, 'test_', fix_bar=True)
# print(ratio_new)