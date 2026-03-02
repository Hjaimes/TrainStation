import torch


def compute_alphas_cumprod(
    num_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
) -> torch.Tensor:
    """Compute alpha_bar schedule (scaled linear beta schedule, matching diffusers)."""
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def build_time_ids(
    original_size: tuple[int, int],
    crop_coords: tuple[int, int],
    target_size: tuple[int, int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build add_time_ids tensor [6] for SDXL conditioning."""
    return torch.tensor([
        original_size[0], original_size[1],
        crop_coords[0], crop_coords[1],
        target_size[0], target_size[1],
    ], dtype=dtype)


def get_velocity(
    latents: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    """Compute v-prediction target: v = sqrt(alpha_bar) * noise - sqrt(1-alpha_bar) * latents."""
    return alpha_bar_t.sqrt() * noise - (1.0 - alpha_bar_t).sqrt() * latents
