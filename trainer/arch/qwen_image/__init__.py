from trainer.registry import register_model
from .strategy import QwenImageStrategy

register_model("qwen_image")(QwenImageStrategy)
