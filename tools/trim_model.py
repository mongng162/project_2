"""
trim_model.py

Trims the MBart vocabulary to only the tokens used by the custom dataset.
This reduces model params from ~1134M to ~120M, enabling T4 GPU training.

Key fix: saves an id_mapping.json so train_slt.py can remap tokenizer IDs.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import torch
import utils
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

import yaml
config_path = 'configs/config_gloss_free.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# ── 1. Load dataset and collect all text ──
train_path = config['data']['train_label_path']
dev_path   = config['data']['dev_label_path']
test_path  = config['data']['test_label_path']

all_text = []
for path in [train_path, dev_path, test_path]:
    if os.path.exists(path):
        data = utils.load_dataset_file(path)
        all_text.extend([v['text'] for v in data.values()])
        print(f"Loaded {len(data)} samples from {path}")

print(f"Total text samples: {len(all_text)}")

# ── 2. Load full tokenizer ──
print("Loading full MBart tokenizer...")
tokenizer = MBartTokenizer.from_pretrained(
    "facebook/mbart-large-cc25",
    src_lang="vi_VN",
    tgt_lang="vi_VN",
    use_fast=False
)

# ── 3. Tokenize ALL text and find used token IDs ──
print("Tokenizing all dataset text...")
used_ids = set()
for text in all_text:
    ids = tokenizer.encode(text, add_special_tokens=True)
    used_ids.update(ids)

# Always include ALL special tokens + language codes
used_ids.update(tokenizer.all_special_ids)
# Add all language code IDs
for lang_code, lang_id in tokenizer.lang_code_to_id.items():
    used_ids.add(lang_id)

used_ids = sorted(used_ids)
print(f"Original vocab: {len(tokenizer)} → Trimmed vocab: {len(used_ids)} tokens")

# ── 4. Create old→new ID mapping ──
old_to_new = {old_id: new_id for new_id, old_id in enumerate(used_ids)}

# Save the mapping for train_slt.py to use
os.makedirs('pretrain_models/MBart_trimmed', exist_ok=True)
mapping_path = 'pretrain_models/MBart_trimmed/id_mapping.json'
with open(mapping_path, 'w') as f:
    json.dump({str(k): v for k, v in old_to_new.items()}, f)
print(f"Saved ID mapping to {mapping_path}")

# ── 5. Load and trim model ──
print("Loading full MBart model...")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

new_vocab_size = len(used_ids)

# Trim embedding weights
old_embed = model.model.shared.weight.data
new_embed = old_embed[used_ids]

model.model.shared = torch.nn.Embedding(new_vocab_size, new_embed.shape[1])
model.model.shared.weight.data = new_embed
model.model.encoder.embed_tokens = model.model.shared
model.model.decoder.embed_tokens = model.model.shared

# Trim LM head
old_lm_head = model.lm_head.weight.data
new_lm_head = old_lm_head[used_ids]
model.lm_head = torch.nn.Linear(new_lm_head.shape[1], new_vocab_size, bias=False)
model.lm_head.weight.data = new_lm_head

# Trim final bias
model.final_logits_bias = model.final_logits_bias[:, used_ids]

# Update config
model.config.vocab_size = new_vocab_size
model.config.tie_word_embeddings = False

print(f"Saving trimmed model to pretrain_models/MBart_trimmed/ ...")
tokenizer.save_pretrained('pretrain_models/MBart_trimmed')
model.save_pretrained('pretrain_models/MBart_trimmed')

# ── 6. Create mytran (visual encoder init) ──
print("Creating mytran model...")
os.makedirs('pretrain_models/mytran', exist_ok=True)
configuration = MBartConfig.from_pretrained('pretrain_models/mytran/config.json')
configuration.vocab_size = new_vocab_size
configuration.tie_word_embeddings = False

mytran_model = MBartForConditionalGeneration._from_config(config=configuration)
mytran_model.model.shared = model.model.shared
mytran_model.save_pretrained('pretrain_models/mytran/')

print(f"\n✅ Done! Trimmed vocab: {new_vocab_size} tokens")
print("  pretrain_models/MBart_trimmed/")
print("  pretrain_models/MBart_trimmed/id_mapping.json")
print("  pretrain_models/mytran/")
