import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from utils import models_info
from hf_runtime import from_pretrained_kwargs

hf_logging.set_verbosity_error()


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
        seed: int | None = None,
    ):
        self.model_name = model_name
        self.use_retrieval = use_retrieval
        self.seed = seed
        self.model_kwargs: dict = model_kwargs or {}
        self.pipeline_kwargs = pipeline_kwargs or {
            "max_new_tokens": 128,
            "num_beams": 4,
            "do_sample": False,
        }
        self._seed_everything()
        self.tokenizer, self.model, self.device = self._initialize_model()

    def _initialize_model(self):
        if "T5" not in self.model_name:
            raise NotImplementedError

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        hf_kwargs = from_pretrained_kwargs()
        tokenizer = AutoTokenizer.from_pretrained(
            models_info[self.model_name]["model_id"],
            **hf_kwargs,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            models_info[self.model_name]["model_id"],
            **hf_kwargs,
            **self.model_kwargs,
        )
        model.to(device)
        model.eval()
        return tokenizer, model, device

    def _seed_everything(self) -> None:
        if self.seed is None:
            return
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def answer_question(self, final_prompt: str) -> str:
        self._seed_everything()
        inputs = self.tokenizer([final_prompt], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.pipeline_kwargs.get("max_new_tokens", 128),
                num_beams=self.pipeline_kwargs.get("num_beams", 4),
                do_sample=self.pipeline_kwargs.get("do_sample", False),
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0].strip()
