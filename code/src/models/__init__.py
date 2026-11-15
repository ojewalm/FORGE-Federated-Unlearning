from .resnet import build_resnet18
from .small_cnn import build_small_cnn
from .text_transformer import build_text_transformer
from .tiny_cnn import build_tiny_cnn


def build_model(name: str, **kwargs):
    if name == "resnet18":
        return build_resnet18(**kwargs)
    if name == "small_cnn":
        return build_small_cnn(**kwargs)
    if name == "text_transformer":
        return build_text_transformer(**kwargs)
    if name == "tiny_cnn":
        return build_tiny_cnn(**kwargs)
    raise ValueError(f"Unknown model: {name}")


__all__ = ["build_model", "build_resnet18", "build_small_cnn", "build_text_transformer", "build_tiny_cnn"]
