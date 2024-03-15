# Better from scratch training

## Motivation
The wishlist contains (1) finding out what are important characteristics of pre-training images and (2) make pre-training (e.g., using simulated data) better for DP fine-tuning because we know what matters thanks to (1)

## Methodology

### Model
Probably want to use FiLM to enable near full adaption possibility for fine-tuning. (For pre-training obviously tuning all parameters.)

### Datasets (pre-training)

- ImageNet-1k (in different variations)

### Datasets (fine-tuning)

- [VTAB](https://arxiv.org/pdf/1910.04867.pdf) (easy to setup and standard benchmark, 19 datasets)
    - from huggingface (2): cifar100 svhn
    - from Tensorflow datasets (easy to download, 8): dmlab dtd eurosat oxford_flowers102 oxford_iiit_pet patch_camelyon resisc45 sun397
    - from Tensorflow datasets (needs additional work):
        - different labels based on task (6 = 3x2): dsprites, clevr, smallnorb
        - manual download (1): diabetic_retinopathy_detection/btgraham-300
        - error (FileFormat.TFRECORD. Got FileFormat.ARRAY_RECORD, 2): caltech101 kitti
- Medical datasets
    - Borja Balle used [CheXpert](https://arxiv.org/pdf/1901.07031.pdf) which is [not on huggingface datasets](https://github.com/huggingface/datasets/issues/6382)
    - Borja Balle also used [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) which is not on huggingface datasets
    - Huggingface: alkzar90/NIH-Chest-X-ray-dataset [paper](https://arxiv.org/pdf/1705.02315.pdf)
    - Huggingface: marmal88/skin_cancer [paper](https://www.nature.com/articles/sdata2018161.pdf)
    - Huggingface: HuggingFaceM4/Stanford-Cars [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Yang_A_Large-Scale_Car_2015_CVPR_paper.pdf)
