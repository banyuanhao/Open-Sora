import argparse
import os
from pathlib import Path
import json
import imageio
from moviepy.editor import ImageSequenceClip, VideoFileClip

import torch
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..",'..'))
from utils.visualization import visualization_utils as vis_utils
from proc_data.scripts.utils import load_pickle, dump_pickle
import cv2

import matplotlib.pyplot as plt

def save_video(video, save_path, fps, save_gif=True, save_mp4=True, use_imageio=False):
    """Save video to a file."""
    # video: [T, C, H, W] or [T, H, W, C], np.ndarray
    if 'float' in str(video.dtype):
        video = np.clip((video * 127.5 + 128), 0, 255).astype('uint8')
    else:
        assert video.dtype == np.uint8, f"unsupported {video.dtype=}"
    if video.shape[1] == 3:
        video = np.moveaxis(video, 1, -1)  # [T, H, W, C]
    if use_imageio:
        assert save_gif and not save_mp4
        imageio.mimwrite(save_path, [img for img in video], fps=fps, loop=10)
        return
    video = ImageSequenceClip([img for img in video], fps=fps)
    mp4_path = save_path.replace('.gif', '.mp4')
    gif_path = save_path.replace('.mp4', '.gif')
    if save_gif:
        video.write_gif(gif_path, verbose=False, logger=None)
    if save_mp4:
        video.write_videofile(mp4_path, verbose=False, logger=None)

def plot_attention_map(self_attention_maps, directory='rope/',fix_bar=True):
    # t axis
    self_attention_map = self_attention_maps.mean(dim=[1,2,4,5])
    torch.save(self_attention_map, os.path.join(directory,'correlation_maps_t.pth'))
    print(self_attention_map.shape)
    # normalize
    self_attention_map = self_attention_map / torch.max(self_attention_map)
    plt.figure(figsize=(10, 10))
    im = plt.imshow(self_attention_map, cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    if fix_bar:
        im.set_clim(0, 1) 
    plt.savefig(directory + 'average_t.png')
    
    # concatenate self_attn_maps with a fliped one
    self_attention_map = torch.cat([self_attention_map.flip(0), self_attention_map,self_attention_map.flip(0)], dim=0)
    # normalize
    self_attention_map = self_attention_map / torch.max(self_attention_map)
    plt.figure(figsize=(10, 20))
    im = plt.imshow(self_attention_map, cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    if fix_bar:
        im.set_clim(0, 1) 
    plt.savefig(directory + 'average_t_.png')
    
    lines = torch.zeros((24,))
    for i in range(24):
        lines += self_attention_map[24+i-12:24+i+12,i]
    lines = lines / 24
    plt.figure(figsize=(10, 10))
    plt.plot(lines)
    # set y axis to 0-1
    plt.ylim(0, 1)
    plt.savefig(directory + 'average_t_line.png')
    
    
    
    # x axis
    self_attention_map = self_attention_maps.mean(dim=[0,1,3,4])
    torch.save(self_attention_map, os.path.join(directory,'correlation_maps_x.pth'))
    print(self_attention_map.shape)
    # normalize
    self_attention_map = self_attention_map / torch.max(self_attention_map)
    plt.figure(figsize=(10, 10))
    im = plt.imshow(self_attention_map, cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    if fix_bar:
        im.set_clim(0, 1) 
    plt.savefig(directory + 'average_x.png')
    
    # concatenate self_attn_maps with a fliped one
    self_attention_map = torch.cat([self_attention_map.flip(0), self_attention_map,self_attention_map.flip(0)], dim=0)
    plt.figure(figsize=(10, 20))
    im = plt.imshow(self_attention_map, cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    if fix_bar:
        im.set_clim(0, 1) 
    plt.savefig(directory + 'average_x_.png')
    
    lines = torch.zeros((16,))
    for i in range(16):
        lines += self_attention_map[16+i-8:16+i+8,i]
    lines = lines / 16
    plt.figure(figsize=(10, 10))
    plt.plot(lines)
    # set y axis to 0-1
    plt.ylim(0, 1)
    plt.savefig(directory + 'average_x_line.png')
    
    
    # y axis
    self_attention_map = self_attention_maps.mean(dim=[0,2,3,5])
    torch.save(self_attention_map, os.path.join(directory,'correlation_maps_y.pth'))
    print(self_attention_map.shape)
    # normalize
    self_attention_map = self_attention_map / torch.max(self_attention_map)
    plt.figure(figsize=(10, 10))
    im = plt.imshow(self_attention_map, cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    if fix_bar:
        im.set_clim(0, 1) 
    plt.savefig(directory + 'average_y.png')
    
    # concatenate self_attn_maps with a fliped one
    self_attention_map = torch.cat([self_attention_map.flip(0), self_attention_map,self_attention_map.flip(0)], dim=0)
    plt.figure(figsize=(10, 20))
    im = plt.imshow(self_attention_map, cmap='jet', aspect='auto')
    cbar = plt.colorbar()
    if fix_bar:
        im.set_clim(0, 1) 
    plt.savefig(directory + 'average_y_.png')
    
    lines = torch.zeros((16,))
    for i in range(16):
        lines += self_attention_map[16+i-8:16+i+8,i]
    lines = lines / 16
    plt.figure(figsize=(10, 10))
    plt.plot(lines)
    # set y axis to 0-1
    plt.ylim(0, 1)
    plt.savefig(directory + 'average_y_line.png')
    
    
    

def visualize_self_attention(self_attention_maps, frame, width, height, fix_bar=True, directory='rope', axis = 'xy', videos = None):
    # select a point in the first several dimensions, and visualize the attention map
    # indicating how much the other pixels are attending to this pixel
    # assert frame width height are list
    assert isinstance(frame, int)
    assert isinstance(width, int)
    assert isinstance(height, int)
    if videos is not None:
        if isinstance(videos, torch.Tensor):
            videos = videos.permute(1,0,2,3).cpu().numpy()
            videos[:, 0,  height*4:height*4+4, width*4:width*4+4] = 0
            videos[:, 1,  height*4:height*4+4, width*4:width*4+4] = 1
            videos[:, 2,  height*4:height*4+4, width*4:width*4+4] = 0
            videos[frame, 0, height*4:height*4+4, width*4:width*4+4] = 1
            videos[frame, 1, height*4:height*4+4, width*4:width*4+4] = 0
            videos[frame, 2, height*4:height*4+4, width*4:width*4+4] = 0
    
    
    self_attention_maps = self_attention_maps[frame, height, width, :, :, :] 
    current_video = self_attention_maps
    frames,widths,heights = current_video.shape
    
    max_value = torch.max(current_video).item()
    if videos is None:
        videos = np.zeros((current_video.shape[0], 3,widths, heights))  
        videos[:, 0,  height, width] = 0
        videos[:, 1,  height, width] = 1
        videos[:, 2,  height, width] = 0
        videos[frame, 0, height, width] = 1
        videos[frame, 1, height, width] = 0
        videos[frame, 2, height, width] = 0
    
    if axis == 'xy':
        pass
    elif axis == 'ty':
        videos = videos.transpose(3, 1, 2, 0)
        current_video = current_video.permute(2,1,0)
    elif axis == 'xt':
        videos = videos.transpose(2, 1, 0, 3)
        current_video = current_video.permute(1,0,2)
    else:
        raise ValueError('axis not supported')
    
    frames = current_video.shape[0]
    
    col = 4
    row = frames // col
    fig, axs = plt.subplots(row, col*2, figsize=(col*5*2,row*5))
    
    for j in range(frames):
        ax_a = axs[j // col, j % col ]
        ax_v = axs[j // col, j % col + col]
        
        video = videos[j]
        video = np.clip((video * 127.5 + 128), 0, 255).astype('uint8')
        video = video/256
        im_v = ax_v.imshow(video.transpose(1,2,0))
        
        
        current_map = current_video[j]
        
        im_a = ax_a.imshow(current_map, cmap='jet', aspect='auto')
        cbar = fig.colorbar(im_a, ax=ax_a)
        if fix_bar:
            im_a.set_clim(0, max_value) 
        
            
        ax_a.axis('off')  # Hide the axes
        ax_v.axis('off') # Hide the axes
        ax_v.set_title(f'Frame {j+1}')
        ax_a.set_title(f'Frame {j+1}')
        
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    path = directory + f'_{axis}_{frame}_{width}_{height}.self.png'
    fig.savefig(path)
    

if __name__ == "__main__":
    # Loads configuration file
    parser = argparse.ArgumentParser()
    token_shape = [24,16,16]
    
    
    parser.add_argument("--path", type=str, help="Path to the pkl file", default='average')
    parser.add_argument("--fix_bar", action='store_true')
    parser.add_argument("--average", type=str, default='rope')
    parser.add_argument("--directory", type=str, default='rope')
    parser.add_argument("--axis", type=str, default='xt')
    parser.add_argument("--video_id", type=int, default=0)
    
    args = parser.parse_args()
    axis = args.axis
    fix_bar = args.fix_bar
    video_id = args.video_id
    
    for i in range(16):
        if args.path == 'average':
            path_names = os.listdir('rope/weights')
            path_names = [path_name for path_name in path_names if path_name.endswith('.pth') and 'maps' in path_name and 'image' not in path_name and 'kubric' not in path_name]
            paths = [os.path.join('rope/weights', path_name) for path_name in path_names]
            data = []
            for path in paths:
                da = torch.load(path).cpu()
                data.append(da)
            data = torch.stack(data, dim=0).mean(dim=0)
            path = 'rope/correlation_maps.pth'
            directory = os.path.join(os.path.dirname(path),os.path.basename(path).split('.')[0])
            video = None
        elif "weights_one" in args.path:
            path = args.path
            dct = torch.load(path)
            data = dct['attn'][video_id]
            summary_text = dct['summary_text'][video_id]
            video = dct['data'][video_id]
            directory = os.path.join(os.path.dirname(path),os.path.basename(path).split('.')[0],str(video_id))
            if not os.path.exists(directory):
                os.mkdir(directory)
            with open(os.path.join(directory,'text.json'),'w') as f:
                json.dump({'text':summary_text},f)
            save_video(video.permute(1,0,2,3).cpu().numpy(),os.path.join(directory,'video.mp4'),fps=12,save_gif=False,save_mp4=True)
            
        else:
            raise ValueError('path not supported')


        data = data.reshape(token_shape + token_shape).cpu()
        # print('data',data.shape)
        plot_attention_map(data)
        # visualize_self_attention(data, 10, 7, i, fix_bar, directory, axis, video)
        break