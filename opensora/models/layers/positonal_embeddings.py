from typing import Dict, Tuple, List, Union
import torch
import torch.nn as nn

class RopePositionalEncoder(nn.Module):
    def __init__(self, encoder_config: Dict, device=None):

        super().__init__()

        self.encoder_config = encoder_config

    def forward(self, q: torch.Tensor, k: torch.Tensor, frames_count: int, height: int, width: int):
        """
        Produces positional encodings for an input of the given size

        :param q: (batch_size, sequence_length, heads, head_dim) The query tensor
        :param k: (batch_size, sequence_length, heads, head_dim) The query tensor
        :param frames_count: The number of frames in the input
        :param height: The height of the input
        :param width: The width of the input
        :return: The positional encodings for the input
        """
        
        raise NotImplementedError()

class VisionLLaMA3DRopePositionalEncoder(RopePositionalEncoder):
    def __init__(self, encoder_config: Dict, device=None):

        super().__init__(encoder_config)

        self.encoder_config = encoder_config
        self.grid_scale = encoder_config.get("grid_scale", (1.0, 1.0, 1.0))
        # The percentage of dimensions that the spatial and temporal components will take
        # Spatial will be multiplied by 2 for height and width
        self.spatial_fraction = encoder_config.get("spatial_fraction", 1/4)
        self.temporal_fraction = encoder_config.get("temporal_fraction", 1/2)

    def make_grid(self, frames_count: int, height: int, width: int, device):
        """
        Produces a grid of the given size

        :param frames_count: The number of frames in the input
        :param height: The height of the input
        :param width: The width of the input
        :return: t_grid (frame_count, height, width), x_grid (frame_count, height, width), y_grid (frame_count, height, width)
        """
        # (frames_count, height, width)
        grid = torch.meshgrid(
            torch.arange(frames_count, device=device) * self.grid_scale[0],
            torch.arange(height, device=device) * self.grid_scale[1],
            torch.arange(width, device=device) * self.grid_scale[2],
            indexing='ij',
        )
        return grid

    def precompute_freqs_cis_3d(self, dim: int, frames_count: int, height: int, width: int, device, theta: Union[float,List[float]] = 10000.0):
        """
        Precomputes frequentcies
        :param dim: the number of features in the tensor that will be embedded
        :param frames_count: The number of frames in the input
        :param height: The height of the input
        :param width: The width of the input
        :param theta: The theta value
        :param device: The device to use
        :return ((frames_count, height, width), dim // 2) The precomputed frequencies
        """
        # Avoids mixed precision computation for encodings
        
        if isinstance(theta, float):
            theta = [theta, theta, theta]
        theta_t, theta_h, theta_w = theta
        with torch.autocast(device_type="cuda", enabled=False):
            # (frames_count, height, width) grids with position for each dimension
            t_pos, h_pos, w_pos = self.make_grid(frames_count, height, width, device)
            t_pos = t_pos.flatten() # N
            h_pos = h_pos.flatten() # N
            w_pos = w_pos.flatten() # N

            t_axis_size = round(dim * self.temporal_fraction)
            s_axis_size = round(dim * self.spatial_fraction)
            if t_axis_size + 2 * s_axis_size != dim:
                raise ValueError(f"Invalid dimensions for the positional encoding. Temporal and spatial dimensions do not sum up to the original dimension. Temporal fraction: {self.temporal_fraction}, Spatial fraction: {self.spatial_fraction}, Temporal dims: {t_axis_size}, Spatial dims: {s_axis_size}, Dim: {dim}")
            if t_axis_size % 2 != 0:
                raise ValueError(f"Invalid dimensions for the positional encoding. Temporal dimension is not divisible by 2. Temporal fraction: {self.temporal_fraction}, Temporal dims: {t_axis_size}, Dim: {dim}")
            if s_axis_size % 2 != 0:
                raise ValueError(f"Invalid dimensions for the positional encoding. Spatial dimension is not divisible by 2. Spatial fraction: {self.spatial_fraction}, Spatial dims: {s_axis_size}, Dim: {dim}")

            t_scale = torch.arange(0, t_axis_size, 2, dtype=torch.float64, device=device) / t_axis_size
            s_scale = torch.arange(0, s_axis_size, 2, dtype=torch.float64, device=device) / s_axis_size

            t_freqs = 1.0 / (theta_t ** t_scale) # Hc/4
            freqs_h = 1.0 / (theta_h ** s_scale) # Hc/8
            freqs_w = 1.0 / (theta_w ** s_scale) # Hc/8
            t_freqs = torch.outer(t_pos, t_freqs).float() # N Hc/4
            w_freqs = torch.outer(w_pos, freqs_w).float() # N Hc/8
            h_freqs = torch.outer(h_pos, freqs_h).float() # N Hc/8
            t_cis = torch.polar(torch.ones_like(t_freqs), t_freqs)
            w_cis = torch.polar(torch.ones_like(w_freqs), w_freqs)
            h_cis = torch.polar(torch.ones_like(h_freqs), h_freqs)
            freqs_cis = torch.cat([t_cis, h_cis, w_cis], dim=-1) # N,Hc/2
        return freqs_cis

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        # x: B N H Hc/2
        # freqs_cis:  N, H*Hc/2 or  N Hc/2
        ndim = x.ndim
        assert 0 <= 1 < ndim

        # frequencies are already of the same shape as x
        if freqs_cis.ndim == x.ndim:
            for shape_freq, shape_x in zip(freqs_cis.shape, x.shape):
                if not shape_freq == shape_x or shape_freq == 1:
                    raise ValueError("Shapes of x ({}) and freqs_cis ({}) are not compatible".format(x.shape, freqs_cis.shape))

        elif freqs_cis.shape[-1] == x.shape[-1]:
            shape = [1 if i == 2 or i == 0 else d for i, d in enumerate(x.shape)]  # 1, N, 1, Hc/2
            freqs_cis = freqs_cis.view(*shape)

        # B, N, H Hc/2
        return freqs_cis

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Avoids mixed precision computation for encodings
        with torch.autocast(device_type="cuda", enabled=False):
            # xq : B N H Hc
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # B N H Hc/2
            if xk is not None:
                xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # B, N, H, Hc
            if xk is not None:
                xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
                xk_out = xk_out.type_as(xk)
            else:
                xk_out = None

            return xq_out.type_as(xq), xk_out

    def forward(self, q: torch.Tensor, k: torch.Tensor, frames_count: int = None, height: int = None, width: int = None, precomputed_freqs_cis_3d: torch.Tensor = None):
        """
        Produces positional encodings for an input of the given size

        :param q: (batch_size, sequence_length, heads, head_dim) The query tensor
        :param k: (batch_size, sequence_length, heads, head_dim) The query tensor
        :param frames_count: The number of frames in the input
        :param height: The height of the input
        :param width: The width of the input
        :param precomputed_freqs_cis_3d: (batch_size, sequence_length, heads, head_dim) The precomputed frequencies
        :return: The positional encodings for the input
        """

        head_dim = q.shape[-1]

        # Avoids mixed precision computation for encodings
        with torch.autocast(device_type="cuda", enabled=False):
            if precomputed_freqs_cis_3d is None:
                precomputed_freqs_cis_3d = self.precompute_freqs_cis_3d(head_dim, frames_count, height, width, device="cuda").to(q.device)
            embedded_queries, embedded_keys = self.apply_rotary_emb(q, k, precomputed_freqs_cis_3d)

        return embedded_queries, embedded_keys
    
    def forward_permute(self, q: torch.Tensor, k: torch.Tensor, frames_count: int = None, height: int = None, width: int = None, precomputed_freqs_cis_3d: torch.Tensor = None):
        """
        Produces positional encodings for an input of the given size

        :param q: (batch_size, sequence_length, heads, head_dim) The query tensor
        :param k: (batch_size, sequence_length, heads, head_dim) The query tensor
        :param frames_count: The number of frames in the input
        :param height: The height of the input
        :param width: The width of the input
        :param precomputed_freqs_cis_3d: (batch_size, sequence_length, heads, head_dim) The precomputed frequencies
        :return: The positional encodings for the input
        """

        head_dim = q.shape[-1]

        # Avoids mixed precision computation for encodings
        with torch.autocast(device_type="cuda", enabled=False):
            if precomputed_freqs_cis_3d is None:
                precomputed_freqs_cis_3d = self.precompute_freqs_cis_3d(head_dim, frames_count, height, width, device="cuda").to(q.device)
            embedded_queries, embedded_keys = self.apply_rotary_emb_permute(q, k, precomputed_freqs_cis_3d)

        return embedded_queries, embedded_keys