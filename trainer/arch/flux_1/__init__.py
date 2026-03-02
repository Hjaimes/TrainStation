from trainer.registry import register_model
from trainer.arch.flux_1.strategy import Flux1Strategy

register_model("flux_1")(Flux1Strategy)
