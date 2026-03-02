from trainer.registry import register_model
from trainer.arch.sd3.strategy import SD3Strategy

register_model("sd3")(SD3Strategy)
