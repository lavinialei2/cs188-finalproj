import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings.
    timesteps: (B,) int64
    returns: (B, dim) float32
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _make_group_norm(num_channels):
    groups = min(8, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = _make_group_norm(out_channels)
        self.norm2 = _make_group_norm(out_channels)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, out_channels * 2)
        self.res_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        emb_out = self.emb_proj(emb)
        scale, shift = emb_out.chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.res_conv(x)


class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


def _match_length(x, target_len):
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] < target_len:
        pad = target_len - x.shape[-1]
        return F.pad(x, (0, pad))
    return x[..., :target_len]


class UNet1D(nn.Module):
    """
    1D Conv U-Net for diffusion over action vectors.
    Input:  x (B, action_dim)
    Cond:   cond (B, state_dim)
    Time:   t (B,)
    Output: noise prediction (B, action_dim)
    """

    def __init__(
        self,
        action_dim,
        cond_dim,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_embed_dim=128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim

        self.in_conv = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            self.down_blocks.append(ResBlock1D(in_ch, out_ch, time_embed_dim))
            in_ch = out_ch
            if i < len(channel_mults) - 1:
                self.downsamples.append(Downsample1D(in_ch))

        self.mid_block = ResBlock1D(in_ch, in_ch, time_embed_dim)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(channel_mults) - 1)):
            out_ch = base_channels * channel_mults[i]
            self.upsamples.append(Upsample1D(in_ch))
            self.up_blocks.append(ResBlock1D(in_ch + out_ch, out_ch, time_embed_dim))
            in_ch = out_ch

        self.out_norm = _make_group_norm(in_ch)
        self.out_conv = nn.Conv1d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        emb = timestep_embedding(t, self.time_embed_dim)
        emb = self.time_mlp(emb) + self.cond_mlp(cond)

        x = self.in_conv(x)
        skips = []
        for i, block in enumerate(self.down_blocks):
            x = block(x, emb)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.mid_block(x, emb)

        skip_levels = skips[:-1]
        for i, block in enumerate(self.up_blocks):
            x = self.upsamples[i](x)
            skip = skip_levels[-(i + 1)]
            x = _match_length(x, skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            x = block(x, emb)

        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        return x.squeeze(1)


class DiffusionScheduler:
    def __init__(self, num_steps, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0
        )

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise):
        return (
            self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x_start
            + self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1) * noise
        )

    def p_sample(self, model, x, t, cond, noise=True):
        beta_t = self.betas[t].unsqueeze(-1)
        alpha_t = self.alphas[t].unsqueeze(-1)
        alpha_bar_t = self.alphas_cumprod[t].unsqueeze(-1)

        pred_noise = model(x, t, cond)
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mean = (x - coef * pred_noise) / torch.sqrt(alpha_t)

        if not noise:
            return mean

        randn = torch.randn_like(x)
        nonzero_mask = (t != 0).float().unsqueeze(-1)
        var = self.posterior_variance[t].unsqueeze(-1)
        return mean + nonzero_mask * torch.sqrt(var) * randn
