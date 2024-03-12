# Better from scratch training

## Motivation
The wishlist contains (1) finding out what are important characteristics of pre-training images and (2) make pre-training (e.g., using simulated data) better for DP fine-tuning because we know what matters thanks to (1)

## Methodology

### Model
Probably want to use FiLM to enable near full adaption possibility for fine-tuning. (For pre-training obviously tuning all parameters.)

### Datasets (pre-training)

- ImageNet-1k (in different variations)

### Datasets (fine-tuning)

- VTAB? (easy to setup and standard benchmark)
- Medical datasets
    - Borja Balle used [CheXpert](https://arxiv.org/pdf/1901.07031.pdf) which is [not on huggingface datasets](https://github.com/huggingface/datasets/issues/6382)
    - Borja Balle also used [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) which is not on huggingface datasets
    - Huggingface: alkzar90/NIH-Chest-X-ray-dataset (https://arxiv.org/pdf/1705.02315.pdf)
    - Huggingface: marmal88/skin_cancer (https://www.nature.com/articles/sdata2018161.pdf)
