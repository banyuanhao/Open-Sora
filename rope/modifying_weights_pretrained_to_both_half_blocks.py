import torch
from collections import OrderedDict

path_source = '/checkpoints/w620_getty_video_dit_rope_selfcond_adamw_bf16_flash_bs1k_16gpus_run_1/step_step=375000.ckpt'
path_target = 'checkpoints/train_getty_concat_rope_b_32f/step_step=40000.ckpt'


checkpoint_source = torch.load(path_source)
keys = checkpoint_source['state_dict'].keys()
for key in keys:
    if 'ema' not in key:
        print(key)