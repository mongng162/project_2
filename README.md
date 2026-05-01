# GFSLT-VLP: Gloss-Free Sign Language Translation with Visual-Language Pretraining

This repository contains my project applying and experimenting with **Gloss-Free Sign Language Translation (GFSLT)** using **Visual-Language Pretraining (VLP)**.

## Origin

The code is based on the official implementation of the ICCV 2023 paper:

> **Gloss-free Sign Language Translation: Improving from Visual-Language Pretraining**
>
> Benjia Zhou, Zhigang Chen, Albert Clapés, Jun Wan, Yanyan Liang, Sergio Escalera, Zhen Lei, Du Zhang
>
> [[Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.html)] [[arXiv](https://arxiv.org/abs/2307.14768)]

Original repository: [zhoubenjia/GFSLT-VLP](https://github.com/zhoubenjia/GFSLT-VLP)

## About This Project

I cloned the original paper's codebase and am adapting it for my own research/project on sign language translation. The goal is to explore gloss-free SLT approaches that do not rely on intermediate gloss annotations, leveraging visual-language pretraining to bridge the semantic gap between sign videos and spoken language text.

## Installation

```bash
conda create -n gfslt python==3.8
conda activate gfslt

# Install PyTorch according to your CUDA version
pip install -r requirements.txt
```

## Usage

### VLP Pretraining

Pretrain the visual encoder using visual-language alignment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1236 --use_env train_vlp.py --batch-size 4 --epochs 80 --opt sgd --lr 0.01 --output_dir out/vlp
```

### VLP Pretraining V2

Jointly pretrain both the visual encoder and text decoder using CLIP + masked self-supervised learning:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1236 --use_env train_vlp_v2.py --batch-size 4 --epochs 80 --opt sgd --lr 0.01 --output_dir out/vlp_v2 --training-refurbish True --noise-rate 0.15 --noise-type omit_last --random-shuffle False
```

### Sign Language Translation

Fine-tune with a pretrained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=1236 --use_env train_slt.py --batch-size 2 --epochs 200 --opt sgd --lr 0.01 --output_dir out/Gloss-Free --finetune ./out/vlp/checkpoint.pth
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1236 --use_env train_slt.py --batch-size 2 --epochs 200 --opt sgd --lr 0.01 --output_dir out/Gloss-Free --resume out/Gloss-Free/best_checkpoint.pth --eval
```

## Pretrained Models

Refer to [pretrain_models/README.md](pretrain_models/README.md) for setting up the MBart weights and GFSLT model files.

## Citation

If you use the original paper's work, please cite:

```bibtex
@InProceedings{Zhou_2023_ICCV,
    author    = {Zhou, Benjia and Chen, Zhigang and Clap\'es, Albert and Wan, Jun and Liang, Yanyan and Escalera, Sergio and Lei, Zhen and Zhang, Du},
    title     = {Gloss-Free Sign Language Translation: Improving from Visual-Language Pretraining},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {20871-20881}
}
```

## License

The original code is released under the MIT license. See [LICENSE](LICENSE) for details.