import torch

# load_checkpoint from path
path = '/checkpoints/w620_getty_video_dit_rope_selfcond_adamw_bf16_flash_bs1k_16gpus_run_1/step_step=375000.ckpt'
# path = 'checkpoints/train_multiple_concat_rope_b_2gpu/step_step=33000.ckpt'
checkpoint = torch.load(path)

# remove the key from the checkpoint
del checkpoint['state_dict']['model.inner_wrapper.model.model.video_image_conditioning_mlp.0.weight']
del checkpoint['state_dict']['model.inner_wrapper.model.model.video_image_conditioning_mlp.0.bias']
del checkpoint['state_dict']['model.inner_wrapper.model.model.dataset_id_conditioning_mlp.0.weight']
del checkpoint['state_dict']['model.inner_wrapper.model.model.dataset_id_conditioning_mlp.0.bias']

del checkpoint['state_dict']['ema_model.inner_wrapper.model.model.video_image_conditioning_mlp.0.weight']
del checkpoint['state_dict']['ema_model.inner_wrapper.model.model.video_image_conditioning_mlp.0.bias']
del checkpoint['state_dict']['ema_model.inner_wrapper.model.model.dataset_id_conditioning_mlp.0.weight']
del checkpoint['state_dict']['ema_model.inner_wrapper.model.model.dataset_id_conditioning_mlp.0.bias']

from collections import OrderedDict

state_dict_new = OrderedDict()

for key in checkpoint['state_dict'].keys():
    if '.attn.' in key:
        state_dict_new[key.replace('.attn.', '.attn_spatial.')] = checkpoint['state_dict'][key]
    else:
        state_dict_new[key] = checkpoint['state_dict'][key]
checkpoint['state_dict'] = state_dict_new
        

path = '/fsx_ori/yban/checkpoints/w620_getty_video_dit_rope_selfcond_adamw_bf16_flash_bs1k_16gpus_run_1/step_step=375000_for_v2.ckpt'
# print(checkpoint['state_dict']['model.inner_wrapper.model.model.video_image_conditioning_mlp.0.weight'].shape)
torch.save(checkpoint, path)


# path = 'checkpoints/train_multiple_concat_unbal_spatial_full_half_blocks/step_step=10000.ckpt'
# # path = 'checkpoints/train_multiple_concat_rope_b_2gpu/step_step=33000.ckpt'
# checkpoint = torch.load(path)
# for key in checkpoint['state_dict'].keys():
#     print(key)