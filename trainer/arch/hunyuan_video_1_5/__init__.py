from trainer.registry import register_model
from .strategy import HunyuanVideo15Strategy

register_model("hunyuan_video_1_5")(HunyuanVideo15Strategy)
