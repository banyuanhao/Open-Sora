if __name__ == '__main__':
    import sys
    sys.path.append('/nfs/yban/large-scale-video-generation')

from contextlib import contextmanager
from torch.nn.attention.flex_attention import create_block_mask
@contextmanager
def set_default_dtype(dtype):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)
from torch.nn.attention.flex_attention import flex_attention
import torch
import torch.nn.functional as F
import math
from rope.motion.utils import plot_rope
from models.vision.layers.positional_encoding.vision_llama_3d_rope_positional_encoder import VisionLLaMA3DRopePositionalEncoder, VisionLLaMA3DRopePositionalEncoder_Mono_ALL


def get_t_x_y(idx, grid_scale = [0.05, 0.2, 0.2]):
    F, H, W = token_shape_image
    return (idx // (W*H)) * grid_scale[0] , (idx % (W*H) // W) *grid_scale[1], (idx % (W*H) % W) *grid_scale[2]

# gaussian attention
def checkerboard(score, batch, head, token_q, token_kv):
    # score = torch.where(torch.abs(token_kv - token_q) % 1 == 0, score * 0.5, score)
    # score = torch.where(torch.abs(token_kv - token_q) % 2 == 0, score * 2.0, score)
    # return score
    dtype = token_q.dtype
    Width_factor = 1
    Height_factor = 1
    decay_factor = 1
    eps = 0.05
    
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [0.05, 0.2, 0.2])
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [0.05, 0.2, 0.2])
    
    Numerator_width = (W_q - W_kv)**2
    Numerator_height = (H_q - H_kv)**2
    
    time_diff = torch.abs(F_q - F_kv) + eps
    Cofficient_time = time_diff
    Denominator_width = 2 * Height_factor**2 * Cofficient_time
    Denominator_height = 2 * Width_factor**2 * Cofficient_time
    Cofficient_height = 1 / torch.sqrt(2 * torch.pi * Height_factor**2 * Cofficient_time)
    
    Cofficient_width = 1 / torch.sqrt(2 * torch.pi * Width_factor**2 * Cofficient_time)
    value = Cofficient_width * torch.exp(-Numerator_width / Denominator_width) * Cofficient_height * torch.exp(-Numerator_height / Denominator_height)
    
    value = torch.where(F_q - F_kv == 0, value*0.56, value)
    
    # plot_rope(16, 8, 8, token_shape, value.mean(0).mean(0), 'rope/motion/gaussian_flexattention', fix_bar=True)
    # # torch.save(value, 'rope/motion/gaussian.pt')
    # value = torch.ones_like(score)
    
    score = score * value
    # score.to(dtype)
    return score

def score_brownian_version_1(batch, head, token_q, token_kv):
    dtype = token_q.dtype
    Width_factor = 1.8
    Height_factor = 1.8
    decay_factor = 1
    eps = 0.005
    
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [0.003, 0.06, 0.06])
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [0.003, 0.06, 0.06])
    
    Numerator_width = (W_q - W_kv)**2
    Numerator_height = (H_q - H_kv)**2
    
    time_diff = torch.abs(F_q - F_kv)
    time_diff = time_diff + eps
    Cofficient_time = time_diff
    Denominator_width = 2 * Height_factor**2 * Cofficient_time
    Denominator_height = 2 * Width_factor**2 * Cofficient_time
    
    Cofficient_height = 1 / torch.sqrt(2 * torch.pi * Height_factor**2 * Cofficient_time)
    Cofficient_width = 1 / torch.sqrt(2 * torch.pi * Width_factor**2 * Cofficient_time)
    
    value = Cofficient_width * torch.exp(-Numerator_width / Denominator_width) * Cofficient_height * torch.exp(-Numerator_height / Denominator_height)

    return value

def score_brownian(batch, head, token_q, token_kv):
    dtype = token_q.dtype
    Width_factor = 1
    Height_factor = 1
    decay_factor = 1
    eps = 0.05
    
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [0.05, 0.2, 0.2])
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [0.05, 0.2, 0.2])
    
    Numerator_width = (W_q - W_kv)**2
    Numerator_height = (H_q - H_kv)**2
    
    time_diff = torch.abs(F_q - F_kv)
    time_diff = time_diff + (time_diff == 0) * 0.1
    Cofficient_time = time_diff
    Denominator_width = 2 * Height_factor**2 * Cofficient_time
    Denominator_height = 2 * Width_factor**2 * Cofficient_time
    Cofficient_height = 1 / torch.sqrt(2 * torch.pi * Height_factor**2 * Cofficient_time)
    
    Cofficient_width = 1 / torch.sqrt(2 * torch.pi * Width_factor**2 * Cofficient_time)
    value = Cofficient_width * torch.exp(-Numerator_width / Denominator_width) * Cofficient_height * torch.exp(-Numerator_height / Denominator_height)
    
    value = torch.where(F_q - F_kv == 0, value*2, value)
    return value

def score_brown(batch, head, token_q, token_kv):
    # score = torch.where(torch.abs(token_kv - token_q) % 1 == 0, score * 0.5, score)
    # score = torch.where(torch.abs(token_kv - token_q) % 2 == 0, score * 2.0, score)
    # return score
    dtype = token_q.dtype
    Width_factor = 1
    Height_factor = 1
    decay_factor = 1
    eps = 0.05
    
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [0.05, 0.2, 0.2])
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [0.05, 0.2, 0.2])
    
    Numerator_width = (W_q - W_kv)**2
    Numerator_height = (H_q - H_kv)**2
    
    time_diff = torch.abs(F_q - F_kv) + eps
    Cofficient_time = time_diff
    Denominator_width = 2 * Height_factor**2 * Cofficient_time
    Denominator_height = 2 * Width_factor**2 * Cofficient_time
    Cofficient_height = 1 / torch.sqrt(2 * torch.pi * Height_factor**2 * Cofficient_time)
    
    Cofficient_width = 1 / torch.sqrt(2 * torch.pi * Width_factor**2 * Cofficient_time)
    value = Cofficient_width * torch.exp(-Numerator_width / Denominator_width) * Cofficient_height * torch.exp(-Numerator_height / Denominator_height)
    
    value = torch.where(F_q - F_kv == 0, value*0.56, value)
    
    return value

def score_gaussian(batch, head, token_q, token_kv):
    dtype = token_q.dtype
    Width_factor = 1
    Height_factor = 1
    decay_factor = 1
    eps = 0.10
    
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [0.007, 0.2, 0.2])
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [0.007, 0.2, 0.2])
    
    Numerator_width = (W_q - W_kv)**2
    Numerator_height = (H_q - H_kv)**2
    
    time_diff = torch.abs(F_q - F_kv) + eps
    Cofficient_time = time_diff
    Denominator_width = 2 * Height_factor**2 * Cofficient_time
    Denominator_height = 2 * Width_factor**2 * Cofficient_time
    Cofficient_height = 1 / torch.sqrt(2 * torch.pi * Height_factor**2 * Cofficient_time)
    
    Cofficient_width = 1 / torch.sqrt(2 * torch.pi * Width_factor**2 * Cofficient_time)
    value = Cofficient_width * torch.exp(-Numerator_width / Denominator_width) * Cofficient_height * torch.exp(-Numerator_height / Denominator_height)
    
    
    return value

def score_scaled_mask_disk_gaussian(batch, head, token_q, token_kv):
    mask = scaled_mask_disk(batch, head, token_q, token_kv)
    value = score_gaussian(batch, head, token_q, token_kv)
    value = value * mask
    return value

def no_operation(score, batch, head, token_q, token_kv):
    return score

# score_disk
with set_default_dtype(torch.float32):
    index = [1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9]
    mapping_index = torch.tensor(index,device='cuda')
    map_index = {i: index[abs(i)] for i in range(-23, 24)}

score_fixed_tensor = torch.randn(6144,6144, device='cuda')
def score_fixed(score, batch, head, token_q, token_kv):
    return score_fixed_tensor * score

def score_disk(score, batch, head, token_q, token_kv):
    # score = torch.where(torch.abs(token_kv - token_q) % 1 == 0, score * 0.5, score)
    # score = torch.where(torch.abs(token_kv - token_q) % 2 == 0, score * 2.0, score)
    # return score
    
    
    F_q, W_q, H_q = get_t_x_y(token_q,grid_scale=[1, 0.2, 0.2])
    F_kv, W_kv, H_kv = get_t_x_y(token_kv,grid_scale=[1, 0.2, 0.2])
    
    # print(((W_q - W_kv)**2 + (H_q - H_kv)**2 <= 2*(F_kv - F_q)**2).shape)
    # value = torch.where((W_q - W_kv)**2 + (H_q - H_kv)**2 <= 2*(F_kv - F_q)**2, 
    #                     score/torch.sum(((W_q - W_kv)**2 + (H_q - H_kv)**2 <= 2*(F_kv - F_q)**2),dim=[1,2]), 0)
    # map each value in F_q-F_kv based on the map_index
    
    F_array = torch.tensor([map_index[i] for i in  (F_q-F_kv).flatten().tolist()], dtype=score.dtype).reshape(F_q.shape).to(F_q.device)
    
    value = torch.where((W_q - W_kv)**2 + (H_q - H_kv)**2 <= 0.2*0.2*2*((F_array))**2,score, 0)
    
    # plot_rope(16, 8, 8, token_shape, value.mean(0).mean(0), 'rope/motion/gaussian_flexattention_disk', fix_bar=True)
    # # torch.save(value, 'rope/motion/gaussian.pt')
    # value = torch.ones_like(score)
    
    score = value
    return score

def causal(batch,head,token_q,token_kv):
    return token_q >= token_kv

index = [1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9]
index_mapping = torch.tensor(index,device='cuda')
index = torch.tensor(list(range(24)),device='cuda')
def mask_disk(batch,head,token_q,token_kv):
    # batch: int for the batch_id
    # head: int for the head_id
    # token_q: torch.tensor of shape F*W*H, F*W*H
    # token_kv: torch.tensor of shape F*W*H, F*W*H
    
    # F_q, W_q, H_q: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for query token index
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [1, 1, 1])
    # F_kv, W_kv, H_kv: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for key token index
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [1, 1, 1])
    
    F_array = F_q - F_kv
    F_array = abs(F_array)
    
    # F_array.unsqueeze(-1) # (F*W*H, F*W*H) -> (F*W*H, F*W*H, 1)
    # F_array = (F_array == index) * index_mapping # (F*W*H, F*W*H, 24) = (F*W*H, F*W*H, 24) *(24)
    # F_array = F_array.sum(-1) # (F*W*H, F*W*H)
    
    F_array = (F_array == 0) * 1 + (F_array == 1) * 2 + (F_array == 2) * 3 + (F_array == 3) * 4 + (F_array == 4) * 4 + (F_array == 5) * 5 + (F_array == 6) * 6 + (F_array == 7) * 6 + (F_array == 8) * 7 + (F_array == 9) * 7 + (F_array == 10) * 7 + (F_array == 11) * 8 + (F_array == 12) * 8 + (F_array == 13) * 8 + (F_array == 14) * 8 + (F_array == 15) * 8 + (F_array == 16) * 8 + (F_array == 17) * 9 + (F_array == 18) * 9 + (F_array == 19) * 9 + (F_array == 20) * 9 + (F_array == 21) * 9 + (F_array == 22) * 9 + (F_array == 23) * 9
    # mapping each element of the F_array to the index
    # F,W,H, F,W,H
    # F_array = torch.tensor([index_mapping[i] for i in  F_array.flatten().tolist()], dtype=token_q.dtype).reshape(F_q.shape).to(F_q.device)
     
    mask = ((W_q - W_kv)**2 + (H_q - H_kv)**2 <= ((F_array))**2)
    return mask

scaled_index = [0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
scaled_index_mapping = torch.tensor(scaled_index,device='cuda')
# scaled_index = torch.tensor(list(range(24)),device='cuda')

def scaled_mask_disk(batch,head,token_q,token_kv):
    # batch: int for the batch_id
    # head: int for the head_id
    # token_q: torch.tensor of shape F*W*H, F*W*H
    # token_kv: torch.tensor of shape F*W*H, F*W*H
    
    # F_q, W_q, H_q: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for query token index
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [1, 1, 1])
    # F_kv, W_kv, H_kv: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for key token index
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [1, 1, 1])
    
    F_array = F_q - F_kv
    F_array = abs(F_array)
    
    # F_array.unsqueeze(-1) # (F*W*H, F*W*H) -> (F*W*H, F*W*H, 1)
    # F_array = (F_array == index) * index_mapping # (F*W*H, F*W*H, 24) = (F*W*H, F*W*H, 24) *(24)
    # F_array = F_array.sum(-1) # (F*W*H, F*W*H)
    
    F_array_scaled = torch.zeros_like(F_array)
    L = F_array.shape[0]
    global scaled_index
    if len(scaled_index) < L:
        # repeat the last element of scaled_index to match the length of F_array
        scaled_index += [scaled_index[-1]] * (L - len(scaled_index))
    for i in range(L):
        F_array_scaled += (F_array == i) * scaled_index[i]
    F_array = F_array_scaled
    # F_array = (F_array == 0) * 1 + (F_array == 1) * 2 + (F_array == 2) * 3 + (F_array == 3) * 4 + (F_array == 4) * 4 + (F_array == 5) * 5 + (F_array == 6) * 6 + (F_array == 7) * 6 + (F_array == 8) * 7 + (F_array == 9) * 7 + (F_array == 10) * 7 + (F_array == 11) * 8 + (F_array == 12) * 8 + (F_array == 13) * 8 + (F_array == 14) * 8 + (F_array == 15) * 8 + (F_array == 16) * 8 + (F_array == 17) * 9 + (F_array == 18) * 9 + (F_array == 19) * 9 + (F_array == 20) * 9 + (F_array == 21) * 9 + (F_array == 22) * 9 + (F_array == 23) * 9
    # mapping each element of the F_array to the index
    # F,W,H, F,W,H
    # F_array = torch.tensor([index_mapping[i] for i in  F_array.flatten().tolist()], dtype=token_q.dtype).reshape(F_q.shape).to(F_q.device)
     
    mask = ((W_q - W_kv)**2 + (H_q - H_kv)**2 <= ((F_array))**2)
    return mask

def scaled_mask_disk_normalize(batch,head,token_q,token_kv):
    # batch: int for the batch_id
    # head: int for the head_id
    # token_q: torch.tensor of shape F*W*H, F*W*H
    # token_kv: torch.tensor of shape F*W*H, F*W*H
    
    # F_q, W_q, H_q: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for query token index
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [1, 1, 1])
    # F_kv, W_kv, H_kv: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for key token index
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [1, 1, 1])
    
    F_array = F_q - F_kv
    F_array = abs(F_array)
    
    # F_array.unsqueeze(-1) # (F*W*H, F*W*H) -> (F*W*H, F*W*H, 1)
    # F_array = (F_array == index) * index_mapping # (F*W*H, F*W*H, 24) = (F*W*H, F*W*H, 24) *(24)
    # F_array = F_array.sum(-1) # (F*W*H, F*W*H)
    
    F_array_scaled = torch.zeros_like(F_array)
    L = F_array.shape[0]
    global scaled_index
    if len(scaled_index) < L:
        # repeat the last element of scaled_index to match the length of F_array
        scaled_index += [scaled_index[-1]] * (L - len(scaled_index))
    scale = torch.zeros_like(F_array)
    for i in range(L):
        F_array_scaled += (F_array == i) * scaled_index[i] 
    F_array = F_array_scaled
    mask = ((W_q - W_kv)**2 + (H_q - H_kv)**2 <= ((F_array))**2)
    mask = mask.float()
    
    return mask


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, score_function = None, mask_function = None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    B = query.size(0)
    H = query.size(1)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool,device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            # print("bool")
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    
    # get the token idx of attn_weight for query and key
    token_q = torch.arange(L, device=query.device,dtype=query.dtype).view(1, 1, -1, 1).expand(B, H, -1, S)
    token_kv = torch.arange(S, device=query.device,dtype=query.dtype).view(1, 1, 1, -1).expand(B, H, L, -1)
    
    if score_function is not None and mask_function is None:
        attn_weight = score_function(attn_weight, batch = batch_size, head=head,  token_q=token_q, token_kv=token_kv)
    elif mask_function is not None and score_function is None:
        mask = mask_function(batch = batch_size, head=head,  token_q=token_q, token_kv=token_kv)
        attn_weight = torch.where(mask, attn_weight, torch.tensor(float("-inf"), device=attn_weight.device, dtype=attn_weight.dtype))
    elif score_function is not None and mask_function is not None:
        raise ValueError("Only one of score_function and mask_function can be used")
    
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    return attn_weight @ value

function_name = 'gaussian'
function_use = score_gaussian
generated_pt = True

fix_bar = True
frame, height, width = 0, 8, 8
token_shape_image = [1,9,16]
token_shape_video = [32,9,16]
token_shape_prod_image = torch.prod(torch.tensor(token_shape_image)).item()
token_shape_prod_video = torch.prod(torch.tensor(token_shape_video)).item()
batch_size = 2
head = 16

# natural batch
query = torch.randn(batch_size,head, token_shape_prod_image, 64, device='cuda',dtype=torch.float32)
key = torch.randn(batch_size, head, token_shape_prod_image, 64, device='cuda', dtype=torch.float32)
value = torch.randn(batch_size, head, token_shape_prod_image, 64, device='cuda', dtype=torch.float32)
target = torch.randn(batch_size, head, token_shape_prod_image, 64, device='cuda', dtype= torch.float32)

L, S = query.size(-2), key.size(-2)
token_q = torch.arange(L, device=query.device,dtype=query.dtype).view(-1, 1).expand(-1, S)
token_kv = torch.arange(S, device=query.device,dtype=query.dtype).view(1, -1).expand(L, -1)
attn_weight = query @ key.transpose(-2, -1) * 1 / math.sqrt(query.size(-1))
tensor_image = function_use(batch = batch_size, head=head,  token_q=token_q, token_kv=token_kv)
print(tensor_image.shape)
tensor_image = tensor_image/torch.max(tensor_image)
plot_rope(0, 8, 8, token_shape_image, tensor_image.cpu(), f'rope/motion/{function_name}_image_{token_shape_prod_image}', fix_bar=True)


# natural batch
query = torch.randn(batch_size,head, token_shape_prod_video, 64, device='cuda',dtype=torch.float32)
key = torch.randn(batch_size, head, token_shape_prod_video, 64, device='cuda', dtype=torch.float32)
value = torch.randn(batch_size, head, token_shape_prod_video, 64, device='cuda', dtype=torch.float32)
target = torch.randn(batch_size, head, token_shape_prod_video, 64, device='cuda', dtype= torch.float32)

L, S = query.size(-2), key.size(-2)
token_q = torch.arange(L, device=query.device,dtype=query.dtype).view(-1, 1).expand(-1, S)
token_kv = torch.arange(S, device=query.device,dtype=query.dtype).view(1, -1).expand(L, -1)
attn_weight = query @ key.transpose(-2, -1) * 1 / math.sqrt(query.size(-1))
tensor_video = function_use(batch = batch_size, head=head,  token_q=token_q, token_kv=token_kv)
print(tensor_video.shape)

tensor_video = tensor_video/torch.max(tensor_video)
plot_rope(0, 8, 8, token_shape_video, tensor_video.cpu(), f'rope/motion/{function_name}_video_{token_shape_video[0]}_{token_shape_video[1]}_{token_shape_video[2]}', fix_bar=True)
save_dict = {
    'image': tensor_image.cpu(),
    'video': tensor_video.cpu()
}
if generated_pt:
    torch.save(save_dict,f'rope/motion/{function_name}_{token_shape_video[0]}_{token_shape_video[1]}_{token_shape_video[2]}.pt')

    print('===========')
    fix_bar = True
    frame, height, width = 0, 8, 8
    batch_size = 2
    head = 16

    rope_encoder =  VisionLLaMA3DRopePositionalEncoder({"grid_scale": [1., 1., 1.]}, 'cuda')
    query = torch.ones(batch_size, token_shape_prod_video, head, 64, device='cuda')
    key = torch.ones(batch_size, token_shape_prod_video, head, 64, device='cuda')

    precompu = rope_encoder.precompute_freqs_cis_3d(64, token_shape_video[0], token_shape_video[1], token_shape_video[2], 'cuda')

    query, key = rope_encoder(query, key, *token_shape_video, precompu)

    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    attn = query @ key.transpose(-2, -1) # B x n_heads x T x T

    # average over the head dimension moayed
    attn = attn.softmax(dim=-1)
    attn = attn.mean(0).mean(0) # S x S
    fix_bar = True
    frame, height, width = 0, 8, 8
    batch_size = 2
    head = 16

    rope_encoder =  VisionLLaMA3DRopePositionalEncoder({"grid_scale": [1., 1., 1.]}, 'cuda')
    query = torch.ones(batch_size, token_shape_prod_image, head, 64, device='cuda')
    key = torch.ones(batch_size, token_shape_prod_image, head, 64, device='cuda')

    precompu = rope_encoder.precompute_freqs_cis_3d(64, token_shape_image[0], token_shape_image[1], token_shape_image[2], 'cuda')

    query, key = rope_encoder(query, key, *token_shape_image, precompu)

    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    attn_ = query @ key.transpose(-2, -1) # B x n_heads x T x T

    # average over the head dimension moayed
    attn_ = attn_.softmax(dim=-1)
    attn_ = attn_.mean(0).mean(0) # S x S

    attn = attn/torch.max(attn)
    attn_ = attn_/torch.max(attn_)

    print(torch.max(attn))
    save_dict_rope = {
        'video': attn.cpu(),
        'image': attn_.cpu()
    }
    torch.save(save_dict_rope,f'rope/motion/rope_{token_shape_video[0]}_{token_shape_video[1]}_{token_shape_video[2]}.pt')

    plot_rope(0, 8, 8, token_shape_video, attn.cpu(), f'rope/motion/rope_video_{token_shape_video[0]}_{token_shape_video[1]}_{token_shape_video[2]}', fix_bar=True)
    plot_rope(0, 8, 8, token_shape_image, attn_.cpu(), f'rope/motion/rope_image_{token_shape_image[0]}_{token_shape_image[1]}_{token_shape_image[2]}', fix_bar=True)

    print('===========')
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [1, 1, 1])
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [1, 1, 1])
    tensor_video_mix = torch.where(F_q == F_kv, attn, tensor_video)
    tensor_image_mix = attn.reshape(*token_shape_video,*token_shape_video)
    tensor_image_mix = tensor_image_mix[0,:,:,0]
    tensor_image_mix = tensor_image_mix.reshape(token_shape_video[-1]*token_shape_video[-2],-1)
    if 'mask' in function_name: 
        tensor_image_mix = tensor_image_mix.bool()
        tensor_video_mix = tensor_video_mix.bool()

    plot_rope(0, 8, 8, token_shape_video, tensor_video_mix.cpu(), f'rope/motion/{function_name}_video_mix_{token_shape_video[0]}_{token_shape_video[1]}_{token_shape_video[2]}', fix_bar=True)
    plot_rope(0, 8, 8, token_shape_image, tensor_image_mix.cpu(), f'rope/motion/{function_name}_image_mix_{token_shape_image[0]}_{token_shape_image[1]}_{token_shape_image[2]}', fix_bar=True)
    # # print the type of the element in the tensor
    # print(tensor_image_mix.dtype)
    # # convert the tensor to bool
    # tensor_image_mix = tensor_image_mix.bool()
    # tensor_video_mix = tensor_video_mix.bool()
    # # print the type of the element in the tensor
    # print(tensor_image_mix.dtype)
    save_dict_mix = {
        'image': tensor_image_mix.cpu(),
        'video': tensor_video_mix.cpu()
    }
    torch.save(save_dict_mix,f'rope/motion/{function_name}_mixed_{token_shape_video[0]}_{token_shape_video[1]}_{token_shape_video[2]}.pt')

    print('===========')

    rope_encoder =  VisionLLaMA3DRopePositionalEncoder_Mono_ALL({}, device = 'cuda')
    query = torch.ones(batch_size, token_shape_prod_video, head, 64, device='cuda')
    key = torch.ones(batch_size, token_shape_prod_video, head, 64, device='cuda')

    precompu = rope_encoder.precompute_freqs_cis_3d(64, token_shape_video[0], token_shape_video[1], token_shape_video[2], 'cuda')

    query, key = rope_encoder(query, key, *token_shape_video, precompu)

    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    attn = query @ key.transpose(-2, -1) # B x n_heads x T x T

    # average over the head dimension moayed
    attn = attn.softmax(dim=-1)
    attn = attn.mean(0).mean(0) # S x S


    rope_encoder =  VisionLLaMA3DRopePositionalEncoder_Mono_ALL({}, device = 'cuda')
    query = torch.ones(batch_size, token_shape_prod_image, head, 64, device='cuda')
    key = torch.ones(batch_size, token_shape_prod_image, head, 64, device='cuda')

    precompu = rope_encoder.precompute_freqs_cis_3d(64, token_shape_image[0], token_shape_image[1], token_shape_image[2], 'cuda')

    query, key = rope_encoder(query, key, *token_shape_image, precompu)

    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    attn_ = query @ key.transpose(-2, -1) # B x n_heads x T x T

    # average over the head dimension moayed
    attn_ = attn_.softmax(dim=-1)
    attn_ = attn_.mean(0).mean(0) # S x S

    attn = attn/torch.max(attn)
    attn_ = attn_/torch.max(attn_)

    print(torch.max(attn))
    save_dict_rope = {
        'video': attn.cpu(),
        'image': attn_.cpu()
    }
    torch.save(save_dict_rope,f'rope/motion/rope_mono_{token_shape_video[0]}_{token_shape_video[1]}_{token_shape_video[2]}.pt')

    plot_rope(0, 8, 8, token_shape_video, attn.cpu(), f'rope/motion/rope_mono_video_{token_shape_video[0]}_{token_shape_video[1]}_{token_shape_video[2]}', fix_bar=True)
    plot_rope(0, 8, 8, token_shape_image, attn_.cpu(), f'rope/motion/rope_mono_image_{token_shape_image[0]}_{token_shape_image[1]}_{token_shape_image[2]}', fix_bar=True)

    print('===========')
