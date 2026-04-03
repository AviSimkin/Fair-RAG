import os
import sys

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
        except ImportError as exc:
            raise ImportError(
                "MLX backend requires mlx and mlx-lm. Install them from requirements.txt."
            ) from exc

        self._generate = generate
        self.model, self.tokenizer = load(models_info[self.model_name]["model_id"])

    def answer_question(self, final_prompt: str) -> str:
        output = self._generate(
            self.model,
            self.tokenizer,
            prompt=final_prompt,
            **self.pipeline_kwargs,
        )
        if isinstance(output, str):
            return output.strip()
        return str(output).strip()
