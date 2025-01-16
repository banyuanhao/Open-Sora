import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_rope(frame, height, width, token_shape, attn, name, fix_bar=True):
    videos = np.zeros((token_shape[0], 3, token_shape[1]*4, token_shape[2]*4))

    frames, _, img_height, img_width = videos.shape
    attn = attn.view(*token_shape, *token_shape)
    # print(attn[16,8,8,15,8,8]/attn[16,8,8,14,8,8])
    # print(attn[16,8,8].sum(dim=1).sum(dim=1))
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
    path = name + f'_{frame}_{width}_{height}_{token_shape}.png'
    fig.savefig(path)
    # close fig
    plt.close(fig)
