import sys
sys.path.append("/nfs/yban/large-scale-video-generation")
import yaml
from omegaconf import OmegaConf
from utils.configuration.yaml_include_loader import IncludeLoader
from dataset.datasets.batched_parallel_iterable_dataset import BatchedParallelIterableDataset
from torch.utils.data import DataLoader
import time
from utils.database.db_connection_manager import DBConnectionManager
from models.vision.layers.opensora.blocks import PatchEmbed3D
import torch
from collections import OrderedDict
import os
import tqdm

# import timm
# # Initialize the pretrained Vision Transformer model (ViT)
# model_name = 'vit_base_patch16_224'  # You can choose different models from timm list
# model = timm.create_model(model_name, pretrained=True)
# model.eval()  # Set model to evaluation mode


def reshape_data(x, frames = 24):
    if len(x.shape) == 4:
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, frames, 1, 1) 
    elif len(x.shape) == 5:
        x = x.permute(0, 2, 1, 3, 4)
    else:
        raise ValueError("Unknown input type of shape {}".format(x.shape))
    return x

def get_model():
    ## Building Model 
    x_embedder = PatchEmbed3D([1,4,4], 3, embed_dim=1024)
    x_embedder_path = 'rope/x_embedder_state_dict.pth'

    # if not os.path.exists(x_embedder_path):
    #     checkpoint_path = "/checkpoints/w620_getty_video_dit_rope_selfcond_adamw_bf16_flash_bs1k_16gpus_run_1/step_step=375000.ckpt"
    #     our_weights = torch.load(checkpoint_path, map_location="cpu")["state_dict"] # Load in CPU memory and only transfer the fully loaded model to GPU to avoid exhausting GPU memory
    #     x_embedder_para = {
    #         'x_embedder.proj.weight': our_weights['ema_model.inner_wrapper.model.model.x_embedder.proj.weight'],
    #         'x_embedder.proj.bias': our_weights['ema_model.inner_wrapper.model.model.x_embedder.proj.bias'],
    #     }
    #     print("Loaded keys: {}".format(our_weights.keys()))
    #     # 将 x_embedder_para 转换为 OrderedDict
    #     x_embedder_state_dict = OrderedDict(x_embedder_para)
    # else:
    x_embedder_state_dict = torch.load(x_embedder_path)

    # 加载权重到 x_embedder
    x_embedder.load_state_dict(x_embedder_state_dict, strict=False)
    x_embedder = x_embedder.to("cuda:0")
    # torch.save(x_embedder_state_dict, "rope/x_embedder_state_dict.pth")
    return x_embedder

model = get_model()

## Loading Dataset
configuration_path = "rope/main_config.yaml"

# Parses the configuration
with open(configuration_path, 'r') as f:
    yaml_object = yaml.load(f, IncludeLoader)

# Loads the configuration file and converts it to a dictionary
omegaconf_config = OmegaConf.create(yaml_object, flags={"allow_objects": True})
configuration = OmegaConf.to_container(omegaconf_config, resolve=True)

# Gets the connection manager
connection_manager_configuration = configuration["connection_manager"]
connection_manager = DBConnectionManager(connection_manager_configuration)
connection = connection_manager.get_new_connection()

connection.close()

dataset_configuration = configuration["dataset"]
transforms_configuration = configuration["transforms"]

dataset = BatchedParallelIterableDataset(dataset_configuration, transforms_configuration, connection_manager)
dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=8, collate_fn=None, pin_memory=True)


with torch.no_grad():
    i = 0
    # j = 0
    tensors = []
    # tensors_image = []
    for current_sample in dataloader:
        current_sample.to("cuda:0")
        if "video" in current_sample.data:
            data = current_sample.data["video"]["video"]
            # print("Video: {}".format(data.shape))
            data = reshape_data(data)
            patch, token_shape = model(data)
            
            attn = patch @ patch.transpose(-2, -1)  # translate attn to float32
            attn = attn.mean(dim=0)

            tensors.append(attn)
            i += 1
            print(i)
            
        # if "image" in current_sample.data:
        #     data = current_sample.data["image"]["video"]
        #     # print("Image: {}".format(data.shape))
        #     data = reshape_data(data)
        #     patch, token_shape = model(data)
            
        #     attn = patch @ patch.transpose(-2, -1)  # translate attn to float32
        #     attn = attn.mean(dim=0)
        #     tensors_image.append(attn)
        #     j += 1
            
            if i%100 == 0 and i != 0:
                maps = torch.stack(tensors, dim=0)
                maps = maps.mean(dim=0)
                torch.save(maps, "rope/correlation_maps_kubric_{}.pth".format(i))
                tensors = []
                
                del maps, patch, attn, data, current_sample  # 删除不再使用的变量
                torch.cuda.empty_cache()  # 清理 CUDA 缓存
                print("Saved correlation maps for batch {}, cache cleared.".format(i))
        
        # if j%100 == 0 and j != 0:
        #     maps = torch.stack(tensors_image, dim=0)
        #     maps = maps.mean(dim=0)
        #     torch.save(maps, "rope/correlation_maps_image_{}.pth".format(j))
        #     tensors_image = []
            
        # print(j)
            

    dataset.close_connection()

    print("All done")