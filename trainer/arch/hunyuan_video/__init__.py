from trainer.registry import register_model
from .strategy import HunyuanVideoStrategy

register_model("hunyuan_video")(HunyuanVideoStrategy)
