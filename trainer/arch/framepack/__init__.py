from trainer.registry import register_model
from .strategy import FramePackStrategy

register_model("framepack")(FramePackStrategy)
