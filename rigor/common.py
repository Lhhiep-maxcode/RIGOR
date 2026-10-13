"""Shared I/O, prompt, and model-loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_model_and_tokenizer(
    model,
    attn_implementation="eager",
    device_map="auto",
    dtype="float16",
    trust_remote_code=True,
    tokenizer=None,
    revision="main",
    local_files_only=False,
):
    """Load a causal language model exactly as required by RIGOR."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer or model,
        trust_remote_code=trust_remote_code,
        revision=revision,
        local_files_only=local_files_only,
    )
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        revision=revision,
        local_files_only=local_files_only,
    )
    return loaded_model, tokenizer


def build_prompt(
    tokenizer,
    system_prompt=None,
    user_prompt=None,
    encode=True,
    add_generation_prompt=True,
    enable_thinking=True,
):
    messages = [
        {"role": "system", "content": system_prompt or "You are a helpful assistant."},
        {"role": "user", "content": user_prompt or "Hi"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=encode,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def torch_dtype(name: str):
    return "auto" if name == "auto" else getattr(torch, name)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(rows: list[dict], path: Path, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or not path.exists() else "a"
    with path.open(mode, encoding="utf-8") as file:
        file.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(row: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
