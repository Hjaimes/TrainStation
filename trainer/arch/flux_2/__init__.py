from trainer.registry import register_model
from .strategy import Flux2Strategy

register_model("flux_2")(Flux2Strategy)
