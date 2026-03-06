import os
import math
import unicodedata
import inspect
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

# File paths and models 
CLOZE_CSV  = 
OUTPUT_CSV = 

MODELS: Dict[str, str] = {
    #"GPT-2": "openai-community/gpt2",
    #"BERT-Large-WWM": "google-bert/bert-large-uncased-whole-word-masking",
    #"Llama-3.2-3B": "meta-llama/Llama-3.2-3B",           # requires Meta HF access
    "Qwen2-7B": "Qwen/Qwen2-7B", # change to actuall qwen model name
    # "DeepSeek-Coder-1.3B": "deepseek-ai/deepseek-coder-1.3b-base",
}

MAX_MEMORY = {0: "20GiB", "cpu": "64GiB"}  

# Some helpers 
def norm_text(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()

def pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def _from_pretrained_dtype_kw():
    # transformers ≥4.44 uses `dtype`; older uses `torch_dtype`
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    return "dtype" if "dtype" in sig.parameters else "torch_dtype"

# Load LLM 
def load_causal(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # pad token helps some hf stacks; use eos/unk if missing
    if tok.pad_token is None:
        tok.pad_token = getattr(tok, "eos_token", tok.unk_token)
    dtype_kw = _from_pretrained_dtype_kw()
    kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "max_memory": MAX_MEMORY if torch.cuda.is_available() else None,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs[dtype_kw] = pick_dtype()
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if hasattr(mdl.config, "use_cache"):
        mdl.config.use_cache = False
    for p in mdl.parameters():
        p.requires_grad_(False)
    return tok, mdl

# Get log probability for a word given a sentence. 
@torch.no_grad()
def causal_phrase_prob(tok, mdl, sentence: str, word: str) -> Tuple[float, float]:
    """
    Exact next-(word|phrase) probability for causal LMs via teacher forcing.
    Returns (raw_prob, log_prob). Supports multi-word phrases.
    """
    sentence = norm_text(sentence)
    word = str(word)  # do NOT normalize/alter the word string per user’s preference

    # ensure exactly one trailing space before generation
    prompt = sentence.rstrip() + " "
    enc = tok(prompt, return_tensors="pt").to(mdl.device)
    input_ids = enc["input_ids"]

    # prefix a space so we score a true word start
    cont_ids = tok(" " + word, add_special_tokens=False).input_ids
    if not cont_ids:
        return float("nan"), float("nan")

    total_logp = 0.0
    for tid in cont_ids:
        out = mdl(input_ids=input_ids)
        probs = F.softmax(out.logits[:, -1, :], dim=-1)[0]
        p = float(probs[tid].clamp_min(1e-45))
        total_logp += math.log(p)
        input_ids = torch.cat([input_ids, torch.tensor([[tid]], device=input_ids.device)], dim=1)

    return math.exp(total_logp), total_logp

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Load cloze data
    df = pd.read_csv(CLOZE_CSV)
    needed = {"sentence", "word", "cloze_prob"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns {needed}")

    # Basic hygiene (don’t alter word strings per your request)
    df["sentence"] = df["sentence"].map(norm_text)
    df["cloze_prob"] = pd.to_numeric(df["cloze_prob"], errors="coerce")

    # Prepare output
    cols = ["llm_name", "sentence", "word", "human_cloze_prob", "llm_next_word_prob", "llm_log_next_word"]
    pd.DataFrame(columns=cols).to_csv(OUTPUT_CSV, index=False)

    def append(rows):
        if rows:
            pd.DataFrame(rows, columns=cols).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    # ---- Causal models on GPU, one at a time ----
    causal_ids = {k: v for k, v in MODELS.items() if k != "BERT-Large-WWM"}
    for llm_name, model_id in causal_ids.items():
        print(f"\n[info] Loading causal model: {llm_name} ({model_id})")
        tok, mdl = load_causal(model_id)

        batch = []
        for r in tqdm(df.itertuples(index=False), total=len(df), desc=f"Scoring {llm_name}", ncols=100):
            sentence = str(r.sentence)
            word     = str(r.word)
            human_p  = float(r.cloze_prob) if np.isfinite(r.cloze_prob) else np.nan
            try:
                p_raw, logp = causal_phrase_prob(tok, mdl, sentence, word)
            except Exception:
                p_raw, logp = float("nan"), float("nan")
            batch.append([llm_name, sentence, word, human_p, p_raw, logp])

            if len(batch) >= 2000:
                append(batch); batch = []
        append(batch)

        # free VRAM
        del tok, mdl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
