from .partition import dirichlet_partition, natural_partition
from .cifar10 import build_cifar10_fcl
from .cifar100 import build_cifar100_fcl
from .tiny_imagenet import build_tiny_imagenet_fcl
from .femnist import build_femnist_fcl
from .newsgroups20 import build_newsgroups20_fcl

DATASET_BUILDERS = {
    "cifar10": build_cifar10_fcl,
    "cifar100": build_cifar100_fcl,
    "tiny_imagenet": build_tiny_imagenet_fcl,
    "femnist": build_femnist_fcl,
    "newsgroups20": build_newsgroups20_fcl,
}


def build_dataset(name: str, **kwargs):
    if name not in DATASET_BUILDERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_BUILDERS)}")
    return DATASET_BUILDERS[name](**kwargs)


__all__ = [
    "build_dataset",
    "dirichlet_partition",
    "natural_partition",
    "DATASET_BUILDERS",
]
