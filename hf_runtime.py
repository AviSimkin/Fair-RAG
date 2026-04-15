"""Helpers for Hugging Face auth and cache runtime configuration."""

from __future__ import annotations

import os
from typing import Dict, Optional

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.realpath(__file__))


def configure_hf_runtime() -> tuple[Optional[str], str]:
    """
    Load .env credentials and establish a stable cache location.

    Returns
    -------
    token     : HF auth token if present
    cache_dir : effective cache directory path
    """
    env_fp = os.path.join(ROOT, ".env")
    if os.path.exists(env_fp):
        load_dotenv(env_fp, override=False)

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    cache_dir = (
        os.getenv("HF_HOME")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or os.getenv("TRANSFORMERS_CACHE")
    )
    if not cache_dir:
        cache_dir = os.path.join(ROOT, ".hf_cache")
        os.environ.setdefault("HF_HOME", cache_dir)

    os.makedirs(cache_dir, exist_ok=True)
    return token, cache_dir


def from_pretrained_kwargs() -> Dict:
    """Return common kwargs for transformers `from_pretrained` calls."""
    token, cache_dir = configure_hf_runtime()
    kwargs: Dict = {"cache_dir": cache_dir}
    if token:
        kwargs["token"] = token
    return kwargs
