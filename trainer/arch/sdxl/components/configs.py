from dataclasses import dataclass


@dataclass(frozen=True)
class SDXLConfig:
    """SDXL variant configuration."""
    name: str
    prediction_type: str           # "epsilon" or "v_prediction"
    latent_channels: int           # 4
    vae_scaling_factor: float      # 0.13025
    num_train_timesteps: int       # 1000
    in_channels: int               # 4
    cross_attention_dim: int       # 2048  (CLIP-L 768 + CLIP-G 1280)
    time_ids_size: int             # 6


SDXL_CONFIGS = {
    "base": SDXLConfig(
        name="sdxl-base-1.0",
        prediction_type="epsilon",
        latent_channels=4,
        vae_scaling_factor=0.13025,
        num_train_timesteps=1000,
        in_channels=4,
        cross_attention_dim=2048,
        time_ids_size=6,
    ),
    "v_pred": SDXLConfig(
        name="sdxl-base-1.0-vpred",
        prediction_type="v_prediction",
        latent_channels=4,
        vae_scaling_factor=0.13025,
        num_train_timesteps=1000,
        in_channels=4,
        cross_attention_dim=2048,
        time_ids_size=6,
    ),
}
