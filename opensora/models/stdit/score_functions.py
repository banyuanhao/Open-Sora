import torch

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