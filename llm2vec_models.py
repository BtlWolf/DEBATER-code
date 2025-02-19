from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import numpy as np
import torch

from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta
from mteb.models.text_formatting_utils import corpus_to_texts

from .instructions import task_to_instruction

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EncodeTypes = Literal["query", "passage"]

class LLM2VecWrapper:
    def __init__(self, *args, **kwargs):
        try:
            from llm2vec import LLM2Vec
        except ImportError:
            raise ImportError(
                "To use the LLM2Vec models `llm2vec` is required. Please install it with `pip install llm2vec`."
            )
        extra_kwargs = {}
        try:
            import flash_attn  # noqa

            extra_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            logger.warning(
                "LLM2Vec models were trained with flash attention enabled. For optimal performance, please install the `flash_attn` package with `pip install flash-attn --no-build-isolation`."
            )
        self.task_to_instructions = None
        if "task_to_instructions" in kwargs:
            self.task_to_instructions = kwargs.pop("task_to_instructions")

        if "device" in kwargs:
            kwargs["device_map"] = kwargs.pop("device")
        elif torch.cuda.device_count() > 1:
            kwargs["device_map"] = None

        self.model = LLM2Vec.from_pretrained(*args, **extra_kwargs, **kwargs)

    def encode(
            self,
            sentences: list[str],
            *,
            prompt_name: str = None,
            **kwargs: Any,  # noqa
    ) -> np.ndarray:
        if prompt_name is not None:
            instruction = (
                self.task_to_instructions[prompt_name]
                if self.task_to_instructions
                   and prompt_name in self.task_to_instructions
                else llm2vec_instruction(task_to_instruction(prompt_name))
            )
        else:
            instruction = ""
        sentences = [[instruction, sentence + "</s>"] for sentence in sentences]
        return self.model.encode(sentences, **kwargs)

    def encode_corpus(
            self,
            corpus: list[dict[str, str]] | dict[str, list[str]] | list[str],
            **kwargs: Any,
    ) -> np.ndarray:
        sentences = corpus_to_texts(corpus, sep=" ")
        sentences = [["",sentence + " Use eight words to represent the above text in multiple aspects: <reserved_12><reserved_13><reserved_14><reserved_15><reserved_16><reserved_17><reserved_18><reserved_19>"]for sentence in sentences]
        if "request_qid" in kwargs:
            kwargs.pop("request_qid")
        return self.model.encode(sentences, **kwargs)

    def encode_queries(self, queries: list[str], **kwargs: Any) -> np.ndarray:
        kwargs['is_q'] = True
        return self.encode(queries, **kwargs)


def _loader(wrapper: type[LLM2VecWrapper], **kwargs) -> Callable[..., Encoder]:
    _kwargs = kwargs

    def loader_inner(**kwargs: Any) -> Encoder:
        return wrapper(**_kwargs, **kwargs)

    return loader_inner



llm2vec_MiniCPM2B = ModelMeta(
    loader=_loader(
        LLM2VecWrapper,
        base_model_name_or_path="xxxxx", # Base MiniCPM Model
        peft_model_name_or_path="xxxxx", # Trained lora parameters
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ),
    name="xxxxxx",  # Custom Name
    languages=["eng_Latn"],
    open_source=True,
    revision=None,
    release_date="2025-01-02",
)

