from trainer.registry import register_model
from .strategy import FluxKontextStrategy

register_model("flux_kontext")(FluxKontextStrategy)
