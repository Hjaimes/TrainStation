class TrainerError(Exception):
    """Base exception for all training framework errors."""
    pass

class ConfigError(TrainerError):
    """Invalid configuration."""
    pass

class ModelLoadError(TrainerError):
    """Failed to load a model component."""
    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(
            f"Failed to load model from '{path}': {reason}\n"
            f"Check that the path exists and the file is a valid model checkpoint.")
