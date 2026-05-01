"""
convert_custom_dataset.py

Converts a custom dataset (checkpoint.json format) to the gzipped pickle format
required by GFSLT-VLP's S2T_Dataset class.

Expected input (checkpoint.json):
[
    {
        "video_id": "D0002",
        "text": "tinh",
        "frame_paths": ["frames/D0002/img001.png", ...]
    },
    ...
]

Output:
  data/custom/labels.train
  data/custom/labels.dev
  data/custom/labels.test
"""

import json
import pickle
import gzip
import os
import sys
import random

# ─────────────────────────────────────────────
# CONFIGURE THESE PATHS FOR YOUR SETUP
# ─────────────────────────────────────────────
CHECKPOINT_JSON = '/kaggle/input/datasets/mongnguynkin/vsl-signlanguage/_output_/data/custom/dataset/checkpoint.json'
FRAMES_BASE_DIR = '/kaggle/input/datasets/mongnguynkin/vsl-signlanguage/_output_/data/custom/frames/'  # root of the frames folder
OUTPUT_DIR      = 'data/custom'

# Train / Dev / Test split ratio
TRAIN_RATIO = 0.8
DEV_RATIO   = 0.1
TEST_RATIO  = 0.1

SEED = 42
# ─────────────────────────────────────────────

def save_dataset(output_path, data_dict):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved {len(data_dict)} samples → {output_path}")


def main():
    print(f"Loading {CHECKPOINT_JSON} ...")
    with open(CHECKPOINT_JSON, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Handle both list and dict formats
    if isinstance(raw, dict):
        samples = list(raw.values())
    else:
        samples = raw

    print(f"Total samples: {len(samples)}")
    print(f"Example sample: {samples[0]}")

    # ── Build the GFSLT-VLP dictionary format ──
    converted = {}
    skipped = 0
    missing_count = 0
    for entry in samples:
        video_id    = str(entry['video_id'])
        text        = str(entry['text'])
        frame_paths = entry['frame_paths']

        relative_paths = []
        for fp in frame_paths:
            # Frame paths may be absolute (e.g. /kaggle/working/data/custom/frames/...)
            # We remap them to be relative to FRAMES_BASE_DIR
            if os.path.isabs(fp):
                # Extract just the video_id/filename part (last 2 components)
                parts = fp.replace('\\', '/').split('/')
                # e.g. ['', 'kaggle', 'working', 'data', 'custom', 'frames', 'D0001N', '000000.jpg']
                # Take the last 2 parts: video_folder/filename
                rel = '/'.join(parts[-2:])
            else:
                rel = fp.lstrip('/')
                if rel.startswith('frames/'):
                    rel = rel[len('frames/'):]
            relative_paths.append(rel)

        converted[video_id] = {
            'name':      video_id,
            'text':      text,
            'imgs_path': relative_paths,
            'length':    len(relative_paths),
        }

    print(f"\nConverted: {len(converted)} | Skipped: {skipped}")
    print(f"\nSample relative path: {list(converted.values())[0]['imgs_path'][0]}")
    print(f"Full path that will be tried: {FRAMES_BASE_DIR}{list(converted.values())[0]['imgs_path'][0]}")
    # Quick sanity check
    test_path = FRAMES_BASE_DIR + list(converted.values())[0]['imgs_path'][0]
    print(f"Path exists? {os.path.exists(test_path)}")

    # ── Split into train / dev / test ──
    keys = list(converted.keys())
    random.seed(SEED)
    random.shuffle(keys)

    n = len(keys)
    n_train = int(n * TRAIN_RATIO)
    n_dev   = int(n * DEV_RATIO)

    train_keys = keys[:n_train]
    dev_keys   = keys[n_train:n_train + n_dev]
    test_keys  = keys[n_train + n_dev:]

    print(f"\nSplit → Train: {len(train_keys)} | Dev: {len(dev_keys)} | Test: {len(test_keys)}")

    train_dict = {k: converted[k] for k in train_keys}
    dev_dict   = {k: converted[k] for k in dev_keys}
    test_dict  = {k: converted[k] for k in test_keys}

    save_dataset(os.path.join(OUTPUT_DIR, 'labels.train'), train_dict)
    save_dataset(os.path.join(OUTPUT_DIR, 'labels.dev'),   dev_dict)
    save_dataset(os.path.join(OUTPUT_DIR, 'labels.test'),  test_dict)

    print("\n✅ Done! Update your config_gloss_free.yaml:")
    print(f"  data:")
    print(f"    train_label_path: ./{OUTPUT_DIR}/labels.train")
    print(f"    dev_label_path:   ./{OUTPUT_DIR}/labels.dev")
    print(f"    test_label_path:  ./{OUTPUT_DIR}/labels.test")
    print(f"    img_path:         {FRAMES_BASE_DIR}")


if __name__ == '__main__':
    main()
