import os
import sys
import gc
from pathlib import Path

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# LangChain's device parameter only accepts integers.
# MPS (Apple Silicon) is targeted via device_map instead.
_DEVICE = 0 if torch.cuda.is_available() else -1
_MPS_AVAILABLE = (
    not torch.cuda.is_available() and torch.backends.mps.is_available()
)

from utils import models_info


class PromptLM:
    """
    model_name (str): model nickname. Can find from utils.complete_model_names
    use_retrieval (bool): Flag indicating whether retrieval-based prompting is used.
    pipeline_kwargs (Dict): Additional arguments to configure the Hugging Face pipeline.
    """

    def __init__(
        self,
        model_name: str,
        use_retrieval: bool = False,
        model_kwargs: dict = None,
        pipeline_kwargs: dict = None,
    ):
        self.model_name = model_name
        self.use_retrieval = use_retrieval
        self.model_kwargs: dict = model_kwargs or {}
        self.pipeline_kwargs = pipeline_kwargs or {
            "max_new_tokens": 128,
            "num_beams": 4,
            "do_sample": False,
        }
        self.tokenizer, self.model = self._initialize_model()

    def _resolve_model_source(self) -> str:
        model_id = models_info[self.model_name]["model_id"]
        direct_path = Path(model_id)
        if direct_path.exists():
            return str(direct_path)

        allow_hf_download = bool(self.model_kwargs.get("allow_hf_download", True))
        try:
            return str(snapshot_download(model_id, local_files_only=True))
        except Exception as exc:
            if allow_hf_download:
                return model_id
            raise FileNotFoundError(
                f"Model {model_id} is not available in local Hugging Face cache. "
                "Download it once online or set allow_hf_download=True."
            ) from exc

    def _initialize_model(self):
        # Flush GPU/CPU memory before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()  # Python garbage collection

        if "T5" not in self.model_name:
            raise NotImplementedError(
                f"PromptLM currently supports T5-family HF models only. Got {self.model_name}"
            )

        model_source = self._resolve_model_source()
        tokenizer_local_only = model_source != models_info[self.model_name]["model_id"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=tokenizer_local_only,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_source,
            local_files_only=tokenizer_local_only,
        )

        if _MPS_AVAILABLE:
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        model.to(device)
        model.eval()
        self._device = device
        return tokenizer, model

    def answer_question(self, final_prompt: str) -> str:
        encoded = self.tokenizer(final_prompt, return_tensors="pt")
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                **self.pipeline_kwargs,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
