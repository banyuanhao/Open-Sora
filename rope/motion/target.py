import sys
sys.path.append('/nfs/yban/large-scale-video-generation')
import torch 
import numpy as np
import matplotlib.pyplot as plt
from rope.motion.utils import plot_rope



token_shape = [24, 16, 16]
grid_scale = [0.05, 0.2, 0.2, 0.05, 0.2, 0.2]


# t, h, w
# i, j, k,      l, m, n

tensor = torch.zeros(token_shape+token_shape)

# generate meshgrid
meshgrid = torch.meshgrid([torch.arange(i) for i in tensor.shape],indexing='ij')
# scale the meshgrid by grid_scale
meshgrid = [meshgrid[i] * grid_scale[i] for i in range(len(meshgrid))]


Width_factor = 1
Height_factor = 1
decay_factor = 1

# t, h, w
# i, j, k,      l, m, n

# decay_factor

# Numerator_width = (k-n)^2
# Numerator_height = (j-m)^2
# time_diff = i-l
# Cofficient_time = (time_diff^2)^(1/decay_factor)
# Denominator_height = 2 * Height_factor^2 * Cofficient_time
# Denominator_width = 2 * Width_factor^2 * Cofficient_time
# Cofficient_height = 1 / sqrt(2 * pi * Height_factor^2 * Cofficient_time)
# Cofficient_width = 1 / sqrt(2 * pi * Width_factor^2 * Cofficient_time)
# value = Cofficient_width * exp(-Numerator_width / Denominator_width) * Cofficient_height * exp(-Numerator_height / Denominator_height)

# calculate the value of the tensor
Numerator_width = (meshgrid[2] - meshgrid[5])**2
Numerator_height = (meshgrid[1] - meshgrid[4])**2

# calcuate meshgrid[0] - meshgrid[3], replace zero element with eps to avoid division by zero
# eps = 0.75 * min(grid_scale)
eps = 0.05
time_diff = meshgrid[0] - meshgrid[3]

# Cofficient_time = torch.pow(time_diff**2, 1/decay_factor)

time_diff = torch.abs(time_diff)
time_diff = time_diff + eps
Cofficient_time = time_diff

Denominator_width = 2 * Height_factor**2 * Cofficient_time
Denominator_height = 2 * Width_factor**2 * Cofficient_time
Cofficient_height = 1 / torch.sqrt(2 * np.pi * Height_factor**2 * Cofficient_time)
Cofficient_width = 1 / torch.sqrt(2 * np.pi * Width_factor**2 * Cofficient_time)
value = Cofficient_width * torch.exp(-Numerator_width / Denominator_width) * Cofficient_height * torch.exp(-Numerator_height / Denominator_height)

# scale the value
scale_factor = torch.max(value[0,:,:,1])
for i in range(token_shape[0]):
    value[i,:,:,i] = value[i,:,:,i] / torch.max(value[i,:,:,i]) * scale_factor * 1.25


plot_rope(16, 8, 8, token_shape, value, 'rope/motion/gaussian', fix_bar=True)
torch.save(value, 'rope/motion/gaussian.pt')

