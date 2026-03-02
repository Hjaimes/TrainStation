from trainer.registry import register_model
from .strategy import ZImageStrategy

register_model("zimage")(ZImageStrategy)
