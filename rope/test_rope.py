import sys
sys.path.append('/nfs/yban/large-scale-video-generation')

from models.vision.layers.positional_encoding.vision_llama_3d_rope_positional_encoder import VisionLLaMA3DRopePositionalEncoder, VisionLLaMA3DRopePositionalEncoder_Disable_Spatial, VisionLLaMA3DRopePositionalEncoder_Disable_Temporal, VisionLLaMA3DRopePositionalEncoder_Mono_ALL, VisionLLaMA3DRopePositionalEncoder_Mono_Spatial, VisionLLaMA3DRopePositionalEncoder_Prod, VisionLLaMA3DRopePositionalEncoder_Real_Prod
import torch
from matplotlib import pyplot as plt
import numpy as np

rope_encoder = VisionLLaMA3DRopePositionalEncoder_Real_Prod({}, 'cuda')
token_shape = (16, 16, 16)
token_shape_prod = torch.prod(torch.tensor(token_shape)).item()

query = torch.ones(1, token_shape_prod, 16, 64, device='cuda')
key = torch.ones(1, token_shape_prod, 16, 64, device='cuda')

precompu = rope_encoder.precompute_freqs_cis_3d(64, token_shape[0], token_shape[1], token_shape[2], 'cuda')
query, key = rope_encoder(query, key, *token_shape, precompu)

# query, key = rope_encoder(query, key, *token_shape, precompu)