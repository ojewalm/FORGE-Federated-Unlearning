"""Dataset preparation entry-point: `python -m src.datasets.prepare --dataset X`.

Idempotent. Downloads and extracts datasets that torchvision/HF/sklearn cannot
fetch on-demand (currently: Tiny-ImageNet).
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


TINY_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def prepare_tiny_imagenet(data_root: Path) -> None:
    target = data_root / "tiny-imagenet-200"
    if target.exists():
        print(f"[tiny_imagenet] already at {target}")
        return
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"[tiny_imagenet] downloading {TINY_URL} ...")
    with urllib.request.urlopen(TINY_URL) as resp:
        buf = io.BytesIO(resp.read())
    print("[tiny_imagenet] extracting ...")
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(data_root)
    print(f"[tiny_imagenet] ready at {target}")


def prepare_cifar100(data_root: Path) -> None:
    from torchvision import datasets
    print("[cifar100] fetching (idempotent) ...")
    datasets.CIFAR100(str(data_root), train=True, download=True)
    datasets.CIFAR100(str(data_root), train=False, download=True)
    print("[cifar100] ready")


def prepare_femnist(data_root: Path) -> None:
    from datasets import load_dataset
    print("[femnist] fetching via HuggingFace flwrlabs/femnist ...")
    load_dataset("flwrlabs/femnist", cache_dir=str(data_root / "hf_cache"))
    print("[femnist] ready")


def prepare_newsgroups20(data_root: Path) -> None:
    from sklearn.datasets import fetch_20newsgroups
    print("[newsgroups20] fetching via sklearn ...")
    fetch_20newsgroups(subset="all", data_home=str(data_root))
    print("[newsgroups20] ready")


PREPARERS = {
    "cifar100": prepare_cifar100,
    "tiny_imagenet": prepare_tiny_imagenet,
    "femnist": prepare_femnist,
    "newsgroups20": prepare_newsgroups20,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PREPARERS.keys()))
    ap.add_argument("--data-root", default="./data")
    args = ap.parse_args()
    PREPARERS[args.dataset](Path(args.data_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
