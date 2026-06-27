"""Local Qwen3-8B backend for proxy ICRL generation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage1.common.chat import apply_chat_template

_MODEL = None
_TOKENIZER = None
_DEVICE = None


def _load_model(model_name: str, dtype: str):
    global _MODEL, _TOKENIZER, _DEVICE
    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    torch_dtype = getattr(torch, dtype)
    _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    _DEVICE = next(_MODEL.parameters()).device
    return _MODEL, _TOKENIZER, _DEVICE


class LocalQwenBackend:
    name = "local_qwen"

    def __init__(
        self,
        *,
        model_name: str | None = None,
        dtype: str = "bfloat16",
        enable_thinking: bool = False,
    ):
        self.model_name = model_name or os.environ.get("ICRL_QWEN_MODEL", "Qwen/Qwen3-8B")
        self.dtype = dtype
        self.enable_thinking = enable_thinking
        self.default_model = self.model_name
        self.judge_model = self.model_name
        _load_model(self.model_name, self.dtype)

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        max_retries: int = 1,
    ) -> str:
        model_obj, tokenizer, device = _load_model(self.model_name, self.dtype)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = apply_chat_template(
            tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = model_obj.generate(**inputs, **gen_kwargs)

        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        max_tokens: int = 256,
    ) -> dict[str, Any]:
        text = self.complete(
            system,
            user,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Expected JSON in response: {text[:200]}")
        return json.loads(match.group())
