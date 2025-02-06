if __name__ == '__main__':
    import sys
    sys.path.append('/home/banyh2000/Open-Sora')

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

# import torch._dynamo

# import os
# # 启用详细日志
# os.environ["TORCH_LOGS"] = "+dynamo"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# # 回退到 Eager 模式
# torch._dynamo.config.suppress_errors = True

def get_t_x_y(idx, grid_scale = [1,1,1]):
    F, W, H = token_shape
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

def score_brownian_version_1(score, batch, head, token_q, token_kv):
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
    
    # value = torch.where(F_q - F_kv == 0, value*2, value)
    
    # plot_rope(0, 8, 8, token_shape, value.mean(0).mean(0), 'rope/motion/brownian_version_1', fix_bar=True)
    # print(value.shape)
    # assert torch.allclose(value[0][0], value[1][2])
    # print(value[0][0].shape)
    # # print the size of value[0][0] in MB
    # print(value[0][0].element_size() * value[0][0].nelement() / 1024 / 1024)
    # torch.save(value[0][0], 'rope/motion/brownian_version_1.pt')
    # value = torch.ones_like(score)
    
    score = score * value
    # score.to(dtype)
    return score

def score_brownian(score, batch, head, token_q, token_kv):
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
    
    # plot_rope(0, 8, 8, token_shape, value.mean(0).mean(0), 'rope/motion/gaussian_flexattention__', fix_bar=True)
    # torch.save(value, 'rope/motion/gaussian.pt')
    # value = torch.ones_like(score)
    
    score = score * value
    # score.to(dtype)
    return score

def score_brown(score, batch, head, token_q, token_kv):
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

def gaussian(score, batch, head, token_q, token_kv):
    dtype = token_q.dtype
    Width_factor = 1
    Height_factor = 1
    decay_factor = 1
    eps = 0.05
    
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale = [0.05, 0.2, 0.2])
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale = [0.05, 0.2, 0.2])
    
    Numerator_width = (W_q - W_kv)**2
    Numerator_height = (H_q - H_kv)**2
    
    time_diff = torch.zeros_like(F_q) + eps * 5
    Cofficient_time = time_diff
    Denominator_width = 2 * Height_factor**2 * Cofficient_time
    Denominator_height = 2 * Width_factor**2 * Cofficient_time
    Cofficient_height = 1 / torch.sqrt(2 * torch.pi * Height_factor**2 * Cofficient_time)
    
    Cofficient_width = 1 / torch.sqrt(2 * torch.pi * Width_factor**2 * Cofficient_time)
    value = Cofficient_width * torch.exp(-Numerator_width / Denominator_width) * Cofficient_height * torch.exp(-Numerator_height / Denominator_height)
    score = score * value
    
    return score

def brown_simplied(score, batch, head, token_q, token_kv):
    Width_factor = 0.2
    Height_factor = 0.2
    F_q, W_q, H_q = get_t_x_y(token_q, grid_scale=[1,1,1])
    F_kv, W_kv, H_kv = get_t_x_y(token_kv, grid_scale=[1,1,1])
    Numerator_width = (W_q - W_kv)**2
    Numerator_height = (H_q - H_kv)**2
    time_diff = torch.abs(F_q - F_kv) * 0.05
    time_diff = time_diff + (time_diff == 0) * 0.05
    Cofficient_time = time_diff
    Denominator_width = 2 * Height_factor**2 * Cofficient_time
    Denominator_height = 2 * Width_factor**2 * Cofficient_time
    Cofficient_height = 1 / torch.sqrt(2 * torch.pi * Height_factor**2 * Cofficient_time)
    Cofficient_width = 1 / torch.sqrt(2 * torch.pi * Width_factor**2 * Cofficient_time)
    value = Cofficient_width * torch.exp(-Numerator_width / Denominator_width) * Cofficient_height * torch.exp(-Numerator_height / Denominator_height)
    score = score * value
    return score


fix_bar = True
frame, height, width = 0, 8, 8
token_shape = [24,16,16]
token_shape_prod = torch.prod(torch.tensor(token_shape)).item()
batch_size = 1
head = 16

# natural batch
query = torch.randn(batch_size,head, token_shape_prod, 64, device='cuda', dtype=torch.float32)
key = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda', dtype=torch.float32)
value = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda', dtype=torch.float32)
target = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda', dtype=torch.float32)

output_standard = scaled_dot_product_attention(query, key, value, score_function=brown_simplied)

output = flex_attention(query, key, value, score_mod=brown_simplied)
torch.testing.assert_close(output, output_standard, atol=2e-2, rtol=2e-2)
print("Pass the uncompile test")

flex_attention = torch.compile(flex_attention)
output_compile = flex_attention(query, key, value, score_mod=brown_simplied)
torch.testing.assert_close(output_standard, output_compile, atol=1e-1, rtol=1e-1)
print("Pass the compile test")

# lightly batch
# query = torch.ones(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
# key = torch.ones(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
# value = torch.ones(8, 8, 2048, 64, device="cuda", dtype=torch.float32)

# query = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
# key = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
# value = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)

# # Call flex_attention with the checkerboard score modification
# flex_attention = torch.compile(flex_attention, dynamic=False)
# # flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
# output_standard = scaled_dot_product_attention(query, key, value, score_function=score_brownian)
# output = flex_attention(query, key, value, score_mod=score_brownian)
# torch.testing.assert_close(output, output_standard, atol=2e-2, rtol=2e-2)

# Testing: repeat 500 times and average the time
# mask_for_test = create_block_mask(mask_disk, batch_size,head, token_shape_prod, token_shape_prod)
# import functools
# causal_attention = functools.partial(flex_attention, block_mask=mask_for_test)
# with set_default_dtype(torch.float16):
#     import time
#     start = time.time()
#     for i in range(10000):
#         query = torch.randn(batch_size,head, token_shape_prod, 64, device='cuda')
#         key = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
#         value = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
#         # output = causal_attention(query, key, value)
#         # output = flex_attention(query, key, value, block_mask=mask_for_test)
#         output = flex_attention(query, key, value, score_mod=score_fixed)
#     end = time.time()
#     print("Time taken for 5000 runs: ", end - start)
#     start = time.time()
#     for i in range(10000):
#         query = torch.randn(batch_size,head, token_shape_prod, 64, device='cuda')
#         key = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
#         value = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
#         with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):  
#             output_standard = F.scaled_dot_product_attention(query, key, value)
#     end = time.time()
#     print("Time taken for 5000 runs: ", end - start)

#     start = time.time()
#     for i in range(5000):
#         query = torch.randn(batch_size,head, token_shape_prod, 64, device='cuda')
#         key = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
#         value = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
#         # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):  
#         output_standard = scaled_dot_product_attention(query, key, value, mask_function=mask_disk)
#     end = time.time()
#     print("Time taken for 5000 runs: ", end - start)

# Testing: compare the output of flex_attention with the standard scaled_dot_product_attention
# output = flex_attention(query, key, value, score_mod=score_brownian)

# with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):  
#     output_standard = scaled_dot_product_attention(query, key, value)
# torch.testing.assert_close(output, output_standard, atol=2e-2, rtol=2e-2)

# Testing: compare the memory of flex_attention with the standard scaled_dot_product_attention
# from memory_profiler import memory_usage
# def run_flex_attention():
#     start = time.time()
#     for i in range(500):
#         output = flex_attention(query, key, value, score_mod=checkerboard)
#     end = time.time()
#     print("Time taken for 500 runs: ", end - start)

# def run_scaled_dot_product_attention():
#     start = time.time()
#     for i in range(500):
#         output_standard = scaled_dot_product_attention(query, key, value)
#     end = time.time()
#     print("Time taken for 500 runs: ", end - start)
    
# # 监控 flex_attention 的内存使用情况
# mem_usage_flex_attention = memory_usage(run_flex_attention)
# print("Memory usage for flex_attention: ", max(mem_usage_flex_attention) - min(mem_usage_flex_attention), "MiB")

# # 监控 scaled_dot_product_attention 的内存使用情况
# mem_usage_scaled_dot_product_attention = memory_usage(run_scaled_dot_product_attention)
# print("Memory usage for scaled_dot_product_attention: ", max(mem_usage_scaled_dot_product_attention) - min(mem_usage_scaled_dot_product_attention), "MiB")

# # test gradient
# mask_for_test = create_block_mask(mask_disk, 8, 8, 2048,2048)
# with set_default_dtype(torch.float32):
#     query = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
#     key = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
#     value = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
#     target = torch.randn(8, 8, 2048, 64, device="cuda", dtype=torch.float32)
#     query.requires_grad_(True)
#     key.requires_grad_(True)
#     value.requires_grad_(True)
#     output_flex = flex_attention(query, key, value, score_mod=score_brownian)
#     # output_flex = scaled_dot_product_attention(query, key, value, mask_function=mask_disk)
#     loss = torch.sum((output_flex-target)**2)
#     loss.backward()
#     # save the gradient of query as the grad_query_flex and make a copy of it
#     grad_query_flex = query.grad.clone()

#     # clean the grad and the loss
#     query.grad.zero_()
#     key.grad.zero_()
#     value.grad.zero_()

#     query.requires_grad_(True)
#     key.requires_grad_(True)
#     value.requires_grad_(True)
#     # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):  
#     output_standard = scaled_dot_product_attention(query, key, value, score_function=score_brownian)
    
#     # L, S = query.size(-2), key.size(-2)
#     # B = query.size(0)
#     # H = query.size(1)
    
#     # token_q = torch.arange(L, device=query.device,dtype=query.dtype).view(1, 1, -1, 1).expand(B, H, -1, S)
#     # token_kv = torch.arange(S, device=query.device,dtype=query.dtype).view(1, 1, 1, -1).expand(B, H, L, -1)
#     # mask = mask_disk(batch = batch_size, head=head,  token_q=token_q, token_kv=token_kv)
#     # output_standard = scaled_dot_product_attention(query, key, value, attn_mask=mask[0][0])
    
#     loss = torch.sum((output_standard-target)**2)
#     loss.backward()
#     grad_query_standard = query.grad.clone()

#     torch.testing.assert_close(output_flex, output_standard, atol=2e-2, rtol=2e-2)
#     torch.testing.assert_close(grad_query_flex, grad_query_standard, atol=2e-2, rtol=2e-2)


# for i in range(5000):
#     query = torch.randn(batch_size,head, token_shape_prod, 64, device='cuda')
#     key = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
#     value = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda') 
#     flex_attention(query, key, value, score_mod=score_brownian)

# flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")
# for i in range(5000):
#     query = torch.randn(batch_size,head, token_shape_prod, 64, device='cuda')
#     key = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
#     value = torch.randn(batch_size, head, token_shape_prod, 64, device='cuda')
#     # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):  
#     # scaled_dot_product_attention(query, key, value)
#     flex_attention(query, key, value, score_mod=score_brownian)