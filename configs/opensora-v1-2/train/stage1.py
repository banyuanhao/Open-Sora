# Dataset settings
dataset = dict(
    type="VariableVideoTextDataset",
    transform_name="resize_crop",
    data_path="video_info.csv",
    num_frames=16,
)

# 修改 batch_size 和采样策略
batch_size = 1
drop_last = True
shuffle = True

# 使用更简单的 bucket 配置
bucket_config = {
    "240p": {  # 320x240 对应 240p
        16: (1.0, 1),   # 16帧，batch_size=1
    }
}

# 减少 worker 数量方便调试
num_workers = 4
num_bucket_build_workers = 4

# backup
# bucket_config = {  # 20s/it
#     "144p": {1: (1.0, 100), 51: (1.0, 30), 102: (1.0, 20), 204: (1.0, 8), 408: (1.0, 4)},
#     # ---
#     "256": {1: (0.5, 100), 51: (0.3, 24), 102: (0.3, 12), 204: (0.3, 4), 408: (0.3, 2)},
#     "240p": {1: (0.5, 100), 51: (0.3, 24), 102: (0.3, 12), 204: (0.3, 4), 408: (0.3, 2)},
#     # ---
#     "360p": {1: (0.5, 60), 51: (0.3, 12), 102: (0.3, 6), 204: (0.3, 2), 408: (0.3, 1)},
#     "512": {1: (0.5, 60), 51: (0.3, 12), 102: (0.3, 6), 204: (0.3, 2), 408: (0.3, 1)},
#     # ---
#     "480p": {1: (0.5, 40), 51: (0.3, 6), 102: (0.3, 3), 204: (0.3, 1), 408: (0.0, None)},
#     # ---
#     "720p": {1: (0.2, 20), 51: (0.3, 2), 102: (0.3, 1), 204: (0.0, None)},
#     "1024": {1: (0.1, 20), 51: (0.3, 2), 102: (0.3, 1), 204: (0.0, None)},
#     # ---
#     "1080p": {1: (0.1, 10)},
#     # ---
#     "2048": {1: (0.1, 5)},
# }

grad_checkpoint = True

# Acceleration settings
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type="STDiT3-XL/2",
    from_pretrained=None,
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=False,
    freeze_y_embedder=True,
)
vae = None
text_encoder = None
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Mask settings
mask_ratios = {
    "random": 0.05,
    "intepolate": 0.005,
    "quarter_random": 0.005,
    "quarter_head": 0.005,
    "quarter_tail": 0.005,
    "quarter_head_tail": 0.005,
    "image_random": 0.025,
    "image_head": 0.05,
    "image_tail": 0.025,
    "image_head_tail": 0.025,
}

# Log settings
seed = 42
outputs = "outputs"
wandb = False
epochs = 1000
log_every = 10
ckpt_every = 200

# optimization settings
load = None
grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 1000

cache_pin_memory = True
pin_memory_cache_pre_alloc_numels = [(290 + 20) * 1024**2] * (2 * 8 + 4)

# 添加一些必要的参数
text_encoder_output_dim = 4096
text_encoder_model_max_length = 300
vae_out_channels = 3

# optimization settings
start_from_scratch = True
