"""
trim_model.py

Prepares MBart model weights for GFSLT-VLP.
Instead of trimming the vocabulary (which causes out-of-bounds errors 
with custom datasets and mismatched sentencepiece models), we simply
prepare the full MBart model and initialize the 'mytran' visual encoder.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import utils
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

print("Loading full MBart tokenizer and model from HuggingFace...")
tokenizer = MBartTokenizer.from_pretrained(
    "facebook/mbart-large-cc25",
    src_lang="vi_VN",  # Set default languages to Vietnamese
    tgt_lang="vi_VN",
    use_fast=False
)

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

# We no longer trim the vocabulary. We keep the full 250,027 tokens.
# This prevents all out-of-bounds errors and sentencepiece mismatch issues!
# Modern GPUs easily have enough VRAM for the full embedding matrix.
vocab_size = model.config.vocab_size
model.config.tie_word_embeddings = False
print(f"Using full vocabulary size: {vocab_size}")

print("Saving MBart tokenizer and model to pretrain_models/MBart_trimmed ...")
os.makedirs('pretrain_models/MBart_trimmed', exist_ok=True)
tokenizer.save_pretrained('pretrain_models/MBart_trimmed')
model.save_pretrained('pretrain_models/MBart_trimmed')

print("Creating mytran model (visual encoder) with full vocab...")
os.makedirs('pretrain_models/mytran', exist_ok=True)
configuration = MBartConfig.from_pretrained('pretrain_models/mytran/config.json')
configuration.vocab_size = vocab_size
configuration.tie_word_embeddings = False

mytran_model = MBartForConditionalGeneration._from_config(config=configuration)
mytran_model.model.shared = model.model.shared
mytran_model.save_pretrained('pretrain_models/mytran/')

print("\nDone! Full models saved to:")
print("  pretrain_models/MBart_trimmed/")
print("  pretrain_models/mytran/")
