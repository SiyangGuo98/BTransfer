# Fine-Tune Once, Reuse Across Models: Bayesian Task-Update Factors and Approximations

This repository provides the official implementation for **Fine-Tune Once, Reuse Across Models: Bayesian Task-Update Factors and Approximations**, accepted as a paper at **ICML 2026**.

The paper formalizes reusable task-update factors for fine-tuning transfer and establishes structural existence results, motivating **BTransfer** as an effective method for reusing fine-tuning knowledge across models without re-training.

## 1. Environment Setup

This repository is developed with **Python 3.11**, **PyTorch 2.6.0 (CUDA 12.4)**, and **Transformers 4.45.2**. To get started, please install the dependencies via:

```bash
conda create -n btransfer python=3.11
conda activate btransfer
pip install -r requirements.txt
```

## 2. Data Preparation

We adopt prompt-based classification and  follow the standard setup used in [LM-BFF](https://github.com/princeton-nlp/LM-BFF) for dataset preparation. To facilitate reproducibility, we additionally provide a lightweight script that streamlines the full workflow via:

```bash
# Download and extract datasets
mkdir -p "cache/datasets/lm-bff/"
wget -P "cache/datasets" https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar
tar xvf "cache/datasets/datasets.tar" -C "cache/datasets/"

# Generate data splits
python scripts/generate.py --seed 42
```

## 3. Model Preparation

Experiments require three categories of models: (i) an old pre-trained model, (ii) its corresponding fine-tuned checkpoint, and (iii) a new pre-trained model. To simplify model download, we provide scripts/download_model.py. Please add new HuggingFace model URLs in constants.py to register additional models.

### 3.1 Old Pre-trained Model

Download the old pre-trained model:
```bash
mkdir -p "cache/models/lm-bff/"
python scripts/download_model.py --model roberta-base
```

### 3.2 Fine-tuned Model

You may fine-tune the model using the workflow described in [Skill-Localization-by-grafting](https://github.com/abhishekpanigrahi1996/Skill-Localization-by-grafting.git), or directly use publicly available checkpoints released by [Localize-and-Stitch](https://github.com/uiuctml/Localize-and-Stitch.git). Place fine-tuned checkpoints under:

```
cache/models/lm-bff/roberta_ckpts/{dataset}-prompt-{model}/
```

### 3.3 New Pre-trained Model

Download the new pre-trained model:
```bash
python scripts/download_model.py --model legal
```

## 4. Running Experiments

### A standard transfer experiment is specified by:
- --model_pt_old: the old pre-trained model
- --model_pt_new: the new pre-trained model
- --fine_tuning_dataset: the fine-tuning dataset
- --transfer_method: the transfer method (options: paramD, TransFusion, BTransfer)

### Example:
```bash
python main.py --model_pt_old roberta-base --model_pt_new legal --fine_tuning_dataset SST-2  --transfer_method paramD
```

## References

This project builds upon the following open-source repositories. We sincerely thank the authors for their contributions:
- [LLM-BFF](https://github.com/princeton-nlp/LM-BFF)
- [Skill-Localization-by-grafting](https://github.com/abhishekpanigrahi1996/Skill-Localization-by-grafting.git)
- [Localize-and-Stitch](https://github.com/uiuctml/Localize-and-Stitch.git)

## Citation

If you find this repository or our work useful, please cite:

```bibtex
@inproceedings{
guo2026finetune,
title={Fine-Tune Once, Reuse Across Models: Bayesian Task-Update Factors and Approximations},
author={Siyang Guo and Junbo Wang and Zibin Zheng},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=iLS4oNpEQ1}
}