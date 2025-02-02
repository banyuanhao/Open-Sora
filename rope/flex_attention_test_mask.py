if __name__ == '__main__':
    import sys
    sys.path.append('/nfs/yban/large-scale-video-generation')

from contextlib import contextmanager
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention
import torch
import torch.nn.functional as F
import math
# from rope.motion.utils import plot_rope

# some setup
# without this, the code for the mask_disk method would run into errors
# however, this is not necessary for the many other mask methods like mask_disk_ori
# maybe the reason is that tinside the mask_disk, we use some more complex torch operations


# import torch._dynamo
# import os
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# torch._dynamo.config.suppress_errors = True

@contextmanager
def set_default_dtype(dtype):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)
        
with set_default_dtype(torch.float32):
    index = [1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9]
    mapping_index = torch.tensor(index,device='cuda')
    map_index = {i: index[abs(i)] for i in range(-23, 24)}

def get_t_x_y(idx, grid_scale = [0.05, 0.2, 0.2]):
    F, W, H = token_shape
    return (idx // (W*H)) * grid_scale[0] , (idx % (W*H) // W) *grid_scale[1], (idx % (W*H) % W) *grid_scale[2]


index = [1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9]
index_mapping = torch.tensor(index,device='cuda')
index = torch.tensor(list(range(24)),device='cuda')

def mask_disk_ori(batch,head,token_q,token_kv):
    # batch: int for the batch_id
    # head: int for the head_id
    # token_q: torch.tensor of shape F*W*H, F*W*H
    # token_kv: torch.tensor of shape F*W*H, F*W*H
    
    # F_q, W_q, H_q: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for query token index
    F_q, W_q, H_q = get_t_x_y(token_q)
    # F_kv, W_kv, H_kv: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for key token index
    F_kv, W_kv, H_kv = get_t_x_y(token_kv)
    
    F_array = F_q - F_kv
    F_array = abs(F_array)
    
    F_array = (F_array == 0) * 1 + (F_array == 1) * 2 + (F_array == 2) * 3 + (F_array == 3) * 4 + (F_array == 4) * 4 + (F_array == 5) * 5 + (F_array == 6) * 6 + (F_array == 7) * 6 + (F_array == 8) * 7 + (F_array == 9) * 7 + (F_array == 10) * 7 + (F_array == 11) * 8 + (F_array == 12) * 8 + (F_array == 13) * 8 + (F_array == 14) * 8 + (F_array == 15) * 8 + (F_array == 16) * 8 + (F_array == 17) * 9 + (F_array == 18) * 9 + (F_array == 19) * 9 + (F_array == 20) * 9 + (F_array == 21) * 9 + (F_array == 22) * 9 + (F_array == 23) * 9
     
    mask = ((W_q - W_kv)**2 + (H_q - H_kv)**2 <= 0.2*0.2*2*((F_array))**2)
    return mask

def mask_disk(batch,head,token_q,token_kv):
    # batch: int for the batch_id
    # head: int for the head_id
    # token_q: torch.tensor of shape F*W*H, F*W*H
    # token_kv: torch.tensor of shape F*W*H, F*W*H
    
    # F_q, W_q, H_q: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for query token index
    F_q, W_q, H_q = get_t_x_y(token_q)
    # F_kv, W_kv, H_kv: torch.tensor of shape (F*W*H, F*W*H), can be viewed as a meshgrid for key token index
    F_kv, W_kv, H_kv = get_t_x_y(token_kv)
    
    F_array = F_q - F_kv
    F_array = abs(F_array)
    
    F_array.unsqueeze(-1) # (F*W*H, F*W*H) -> (F*W*H, F*W*H, 1)
    F_array = (F_array == index) * index_mapping # (F*W*H, F*W*H, 24) = (F*W*H, F*W*H, 24) *(24)
    F_array = F_array.sum(-1) # (F*W*H, F*W*H)
    
    # F_array = (F_array == 0) * 1 + (F_array == 1) * 2 + (F_array == 2) * 3 + (F_array == 3) * 4 + (F_array == 4) * 4 + (F_array == 5) * 5 + (F_array == 6) * 6 + (F_array == 7) * 6 + (F_array == 8) * 7 + (F_array == 9) * 7 + (F_array == 10) * 7 + (F_array == 11) * 8 + (F_array == 12) * 8 + (F_array == 13) * 8 + (F_array == 14) * 8 + (F_array == 15) * 8 + (F_array == 16) * 8 + (F_array == 17) * 9 + (F_array == 18) * 9 + (F_array == 19) * 9 + (F_array == 20) * 9 + (F_array == 21) * 9 + (F_array == 22) * 9 + (F_array == 23) * 9
    # mapping each element of the F_array to the index
    # F,W,H, F,W,H
    # F_array = torch.tensor([index_mapping[i] for i in  F_array.flatten().tolist()], dtype=token_q.dtype).reshape(F_q.shape).to(F_q.device)
     
    mask = ((W_q - W_kv)**2 + (H_q - H_kv)**2 <= 0.2*0.2*2*((F_array))**2)
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


fix_bar = True
frame, height, width = 12, 8, 8
token_shape = [24,16,16]
token_shape_prod = torch.prod(torch.tensor(token_shape)).item()
batch_size = 1
head = 16

# Call flex_attention with the checkerboard score modification
# flex_attention = torch.compile(flex_attention, dynamic=False)
flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
# generating masks, we use the mask_disk_ori/mask_disk for the test
mask_for_test = create_block_mask(mask_disk, batch_size,head, token_shape_prod, token_shape_prod)
import functools
mask_attn = functools.partial(flex_attention, block_mask=mask_for_test)



# Testing: repeat 10000 times and average the time
with set_default_dtype(torch.float16):
    import time
    start = time.time()
    for i in range(10000):
        query = torch.randn(batch_size,head, token_shape_prod, 64, device='cuda')
        key = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
        value = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
        output = mask_attn(query, key, value)
        # output = flex_attention(query, key, value, block_mask=mask_for_test)
        # output = flex_attention(query, key, value, score_mod=score_fixed)
    end = time.time()
    print("Time taken for 10000 runs: ", end - start)
    start = time.time()
    for i in range(10000):
        query = torch.randn(batch_size,head, token_shape_prod, 64, device='cuda')
        key = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
        value = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):  
            output_standard = F.scaled_dot_product_attention(query, key, value)
    end = time.time()
    print("Time taken for 10000 runs: ", end - start)