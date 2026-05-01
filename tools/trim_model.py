"""
trim_model.py

Prepares trimmed MBart tokenizer and model weights for the GFSLT-VLP project.
Rewrote to avoid hftrim which is incompatible with newer transformers versions.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import utils
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

print("Loading dataset...")
raw_data = utils.load_dataset_file('data/Phonexi-2014T/labels.train')
sentences = [v['text'] for v in raw_data.values()]
print(f"Loaded {len(sentences)} training sentences.")

print("Loading tokenizer from HuggingFace (this may download ~5GB model weights)...")
tokenizer = MBartTokenizer.from_pretrained(
    "facebook/mbart-large-cc25",
    src_lang="de_DE",
    tgt_lang="de_DE",
    use_fast=False
)

print("Building trimmed vocabulary from dataset...")
# Tokenize all sentences and collect unique token IDs
used_ids = set()
for sent in sentences:
    ids = tokenizer.encode(sent, add_special_tokens=True)
    used_ids.update(ids)

# Always keep all special tokens
special_ids = set(tokenizer.all_special_ids)
used_ids.update(special_ids)
used_ids = sorted(used_ids)
print(f"Vocabulary size: {len(tokenizer)} -> trimmed to {len(used_ids)} tokens")

print("Loading full MBart model...")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

# Build a mapping from old vocab IDs to new trimmed IDs
old_to_new = {old_id: new_id for new_id, old_id in enumerate(used_ids)}
new_vocab_size = len(used_ids)

# Trim the embedding weights
old_embed = model.model.shared.weight.data
new_embed = old_embed[used_ids]  # shape: [new_vocab_size, hidden_size]

# Re-initialize shared embedding with trimmed weights
model.model.shared = torch.nn.Embedding(new_vocab_size, new_embed.shape[1])
model.model.shared.weight.data = new_embed

# Tie encoder/decoder embeddings to the same trimmed table
model.model.encoder.embed_tokens = model.model.shared
model.model.decoder.embed_tokens = model.model.shared

# Trim the LM head (output projection)
old_lm_head = model.lm_head.weight.data
new_lm_head = old_lm_head[used_ids]
model.lm_head = torch.nn.Linear(new_lm_head.shape[1], new_vocab_size, bias=False)
model.lm_head.weight.data = new_lm_head

# Trim the final bias
model.final_logits_bias = model.final_logits_bias[:, used_ids]

# Update config
model.config.vocab_size = new_vocab_size
model.config.tie_word_embeddings = False

print("Saving trimmed tokenizer and model to pretrain_models/MBart_trimmed ...")
os.makedirs('pretrain_models/MBart_trimmed', exist_ok=True)
tokenizer.save_pretrained('pretrain_models/MBart_trimmed')
model.save_pretrained('pretrain_models/MBart_trimmed')

print("Creating mytran model (visual encoder) from trimmed vocab...")
os.makedirs('pretrain_models/mytran', exist_ok=True)
configuration = MBartConfig.from_pretrained('pretrain_models/mytran/config.json')
configuration.vocab_size = new_vocab_size
configuration.tie_word_embeddings = False

mytran_model = MBartForConditionalGeneration._from_config(config=configuration)
mytran_model.model.shared = model.model.shared
mytran_model.save_pretrained('pretrain_models/mytran/')

print("\nDone! Models saved to:")
print("  pretrain_models/MBart_trimmed/")
print("  pretrain_models/mytran/")
