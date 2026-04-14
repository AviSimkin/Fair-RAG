import os
import sys
from pathlib import Path
import json

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

from utils import models_info


class PromptLMMLX:
    """Minimal MLX-backed generator interface compatible with experiment.py."""

    def __init__(
        self,
        model_name: str,
        use_retrieval: bool = False,
        model_kwargs: dict | None = None,
        pipeline_kwargs: dict | None = None,
    ):
        self.model_name = model_name
        self.use_retrieval = use_retrieval
        self.model_kwargs = model_kwargs or {}
        self.pipeline_kwargs = pipeline_kwargs or {
            "max_tokens": 128,
            "verbose": False,
        }
        try:
            from mlx_lm import load, generate
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "MLX backend requires mlx and mlx-lm. Install them from requirements.txt."
            ) from exc

        self._generate = generate
        self._snapshot_download = snapshot_download
        model_id = models_info[self.model_name]["model_id"]
        allow_hf_download = bool(self.model_kwargs.get("allow_hf_download", False))

        model_path = self._resolve_model_path(
            model_id=model_id,
            allow_hf_download=allow_hf_download,
        )
        self._validate_quantization_compatibility(model_path)
        self.model, self.tokenizer = load(str(model_path))

    def _resolve_model_path(self, model_id: str, allow_hf_download: bool) -> Path:
        """Resolve model path from local HF cache, optionally downloading once."""
        direct_path = Path(model_id)
        if direct_path.exists():
            return direct_path

        if allow_hf_download:
            return Path(self._snapshot_download(model_id))

        try:
            return Path(self._snapshot_download(model_id, local_files_only=True))
        except Exception as exc:
            raise FileNotFoundError(
                f"Model {model_id} is not available in the local Hugging Face cache. "
                "Download it once (online) or set allow_hf_download=True for this run."
            ) from exc

    def _validate_quantization_compatibility(self, model_path: Path) -> None:
        """Fail fast with a clear message when model quantization is unsupported by MLX."""
        config_fp = model_path / "config.json"
        if not config_fp.exists():
            return

        try:
            with open(config_fp, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            return

        quant_cfg = config.get("quantization") or {}
        bits = quant_cfg.get("bits", None)
        if bits is None:
            return

        supported_bits = {2, 3, 4, 5, 6, 8}
        if int(bits) not in supported_bits:
            raise ValueError(
                f"Model quantization bits={bits} is not supported by current MLX runtime. "
                "Supported bits: 2, 3, 4, 5, 6, 8. "
                "Use a compatible MLX model/quantization or upgrade to a runtime that supports this bit width."
            )

    def answer_question(self, final_prompt: str) -> str:
        # Apply chat template if the tokenizer has one, so instruction-tuned
        # models (e.g. LFM 2.5) receive properly formatted input.
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": final_prompt}]
            # Disable chain-of-thought thinking for models that support the flag
            # (e.g. Qwen3). Falls back silently if the template doesn't accept it.
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            prompt = final_prompt
        output = self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            **self.pipeline_kwargs,
        )
        raw = output.strip() if isinstance(output, str) else str(output).strip()
        return self._clean_output(raw)

    @staticmethod
    def _clean_output(text: str) -> str:
        """Strip common formatting artifacts produced by instruction-tuned MLX models.

        Handles:
        - <think>…</think> reasoning blocks (Qwen3 thinking mode)
        - Plain-text reasoning preambles ("Thinking Process: …") followed by an answer
        - **Headline:** / "Headline:" label prefix (LFM, Qwen)
        - Markdown bold markers (**…**)
        - Surrounding quotation marks
        """
        import re

        # 1. Remove <think>…</think> blocks (Qwen3 XML-style chain-of-thought)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 2. If response starts with a reasoning preamble that ends with a labelled
        #    answer line (e.g. "**Headline:** …"), extract only that final part.
        #    Pattern: optional "Thinking Process / Thought / Reasoning" block,
        #    then a recognisable answer marker on its own line.
        answer_marker = re.search(
            r"(?:^|\n)\s*(?:\*{0,2}(?:headline|answer|output|result):?\*{0,2})\s*[:\-]?\s*(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if answer_marker:
            text = answer_marker.group(1).strip()

        # 3. Strip leading "**Headline:**" / "Headline:" label if still present
        text = re.sub(r"^\*{0,2}(?:headline|answer|output|result)\*{0,2}\s*:?\s*", "", text, flags=re.IGNORECASE).strip()

        # 4. Strip surrounding markdown bold (**…** or *…*)
        text = re.sub(r"^\*{1,2}(.*?)\*{1,2}$", r"\1", text, flags=re.DOTALL).strip()

        # 5. Strip surrounding quotation marks (straight or curly)
        if len(text) >= 2 and text[0] in ('"', "'", "\u201c", "\u2018") and text[-1] in ('"', "'", "\u201d", "\u2019"):
            text = text[1:-1].strip()

        return text
