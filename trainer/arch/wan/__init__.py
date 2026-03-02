from trainer.registry import register_model
from .strategy import WanStrategy

register_model("wan")(WanStrategy)
