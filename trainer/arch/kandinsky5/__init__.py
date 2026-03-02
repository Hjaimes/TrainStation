from trainer.registry import register_model
from .strategy import Kandinsky5Strategy

register_model("kandinsky5")(Kandinsky5Strategy)
