from trainer.registry import register_model
from trainer.arch.sdxl.strategy import SDXLStrategy

register_model("sdxl")(SDXLStrategy)
