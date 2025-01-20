import torch
import torch.nn as nn
import torch.nn.attention.flex_attention
# torch.set_float32_matmul_precision('high')
token_shape = [24, 16, 16]
def get_t_x_y(idx, grid_scale = [1, 1, 1]):
    F, W, H = token_shape
    return (idx // (W*H)) * grid_scale[0] , (idx % (W*H) // W) *grid_scale[1], (idx % (W*H) % W) *grid_scale[2]

def score_brownian(score, batch, head, token_q, token_kv, T, W, H, grid_scale = [1, 1, 1]):
    Width_factor = 0.2
    Height_factor = 0.2    
    F_q, W_q, H_q = (token_q // (W*H)) * grid_scale[0] ,(token_q % (W*H) // W) *grid_scale[1], (token_q % (W*H) % W) * grid_scale[2]
    F_kv, W_kv, H_kv = (token_kv // (W*H)) * grid_scale[0] , (token_kv % (W*H) // W) *grid_scale[1], (token_kv % (W*H) % W) * grid_scale[2]
    
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
    score = score + value
    
    return score

def score_test(score, batch, head, token_q, token_kv):
    score = torch.where(token_q//2 == 0, score*2, score)
    return score

import functools
score_brownian = functools.partial(score_brownian, T=24, H=16, W=16, grid_scale=[1, 1, 1])

class Repro(nn.Module):
    def __init__(self, n_head=4):
        super().__init__()
        self.qkv_proj = nn.Linear(n_head * 64, n_head * 64 * 3)
        self.n_head = n_head
        self.d_attn = n_head * 64
        self.qkv_proj.weight.data.fill_(0.1)
        self.qkv_proj.bias.data.fill_(0.1)

    def forward(self, x):
        n_batch, n_ctx, _ = x.shape
        q, k, v = self.qkv_proj(x).split([self.d_attn, self.d_attn, self.d_attn], dim=2)
        q = q.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        k = k.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        v = v.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        return torch.nn.attention.flex_attention.flex_attention(q, k, v, score_brownian)

torch.set_default_device("cuda")
torch.manual_seed(0)

n_head = 16
model = Repro(n_head=n_head)
compiled_model = Repro(n_head=n_head)
compiled_model = torch.compile(compiled_model)

# token_shape = [24,16,16]
# token_shape_prod = torch.prod(torch.tensor(token_shape)).item()
# batch_size = 1
# head = 16
# x = torch.randn(batch_size, token_shape_prod, 64*head, device='cuda',dtype=torch.float32, requires_grad=True)
# x_compiled = x.clone().detach().requires_grad_(True)


x = torch.randn((1, 6144, 64*n_head), requires_grad=True)
x_compiled = x.clone().detach().requires_grad_(True)

out = model(x)
out_compiled = compiled_model(x_compiled)

out.sum().backward()
out_compiled.sum().backward()

torch.testing.assert_close(out, out_compiled, atol=2e-2, rtol=2e-2)

torch.testing.assert_close(model.qkv_proj.weight.grad, compiled_model.qkv_proj.weight.grad, atol=2e-2, rtol=2e-2)

# weight_diff = torch.max(torch.abs(model.qkv_proj.weight.grad - compiled_model.qkv_proj.weight.grad)).item()
# bias_diff = torch.max(torch.abs(model.qkv_proj.bias.grad - compiled_model.qkv_proj.bias.grad)).item()

# print(f"Weight grad max abs diff: {weight_diff:.2e}")
# print(f"Bias grad max abs diff: {bias_diff:.2e}")