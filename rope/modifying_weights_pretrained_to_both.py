import torch
from collections import OrderedDict

path_source = '/checkpoints/w620_getty_video_dit_rope_selfcond_adamw_bf16_flash_bs1k_16gpus_run_1/step_step=375000.ckpt'
path_target = 'checkpoints/train_getty_concat_rope_b_32f/step_step=40000.ckpt'


checkpoint_source = torch.load(path_source)
checkpoint_target = torch.load(path_target)


keys_set_target = set(checkpoint_target['state_dict'].keys())
del checkpoint_source['state_dict']['model.inner_wrapper.model.model.video_image_conditioning_mlp.0.weight']
del checkpoint_source['state_dict']['model.inner_wrapper.model.model.video_image_conditioning_mlp.0.bias']
del checkpoint_source['state_dict']['model.inner_wrapper.model.model.dataset_id_conditioning_mlp.0.weight']
del checkpoint_source['state_dict']['model.inner_wrapper.model.model.dataset_id_conditioning_mlp.0.bias']

del checkpoint_source['state_dict']['ema_model.inner_wrapper.model.model.video_image_conditioning_mlp.0.weight']
del checkpoint_source['state_dict']['ema_model.inner_wrapper.model.model.video_image_conditioning_mlp.0.bias']
del checkpoint_source['state_dict']['ema_model.inner_wrapper.model.model.dataset_id_conditioning_mlp.0.weight']
del checkpoint_source['state_dict']['ema_model.inner_wrapper.model.model.dataset_id_conditioning_mlp.0.bias']

state_dict_new = OrderedDict()

for key in checkpoint_source['state_dict'].keys():
    if '.attn.' in key:
        state_dict_new[key.replace('.attn.', '.attn_spatial.')] = checkpoint_source['state_dict'][key]
        state_dict_new[key.replace('.attn.', '.attn_temporal.')] = checkpoint_source['state_dict'][key]
    else:
        state_dict_new[key] = checkpoint_source['state_dict'][key]
        
checkpoint_source['state_dict'] = state_dict_new

path = '/fsx_ori/yban/checkpoints/w620_getty_video_dit_rope_selfcond_adamw_bf16_flash_bs1k_16gpus_run_1/step_step=375000_for_v2_spa_and_full.ckpt'
# print(checkpoint['state_dict']['model.inner_wrapper.model.model.video_image_conditioning_mlp.0.weight'].shape)
torch.save(checkpoint_source, path)
        
keys_set_new = set(state_dict_new.keys())

diff = keys_set_new - keys_set_target
for key in diff:
    if 'adaLN' not in key:
        print(key)
diff = keys_set_target - keys_set_new
for key in diff:
    if 'conditioning' not in key:
        print(key)