# PEFT Methods in Differentially Private Deep Learning

| Run script | Raw data |
|------------|----------|
| [Run script](../experiments/01-film-vs-lora-evaluation/scripts/run.sh) | [Raw data](../experiments/01-film-vs-lora-evaluation/data.zip) |
| [Run script - Extension 50 Trials, 10% Subset](../experiments/01-film-vs-lora-evaluation__Extension_Subset0.1_Trials50/scripts/run.sh) | [Raw data](../experiments/01-film-vs-lora-evaluation__Extension_Subset0.1_Trials50/data.zip) |
| [Run script - Extension Full batch, 10% Subset](../experiments/01-film-vs-lora-evaluation__Extension_Subset0.1_FullBatch/scripts/run.sh) | [Raw data](../experiments/01-film-vs-lora-evaluation__Extension_Subset0.1_FullBatch/data.zip) |

## Objective
We aim to evaluate the effectiveness of different Parameter Efficient Fine-tuning (PEFT) methods—FiLM and LoRA—in the context of differentially private deep learning to determine the most suitable method for our future experiments.

## Methodology

- **PEFT Methods**: We will compare two PEFT methods—FiLM and LoRA.
- **Hyperparameter Optimization**: We will use Bayesian optimization to fine-tune all hyperparameters, including batch size, epochs, learning rate, and max gradient norm, for each PEFT method.
- **Epsilon Values**: We will conduct experiments for each PEFT method with differential privacy guarantees at epsilon values of 0.25, 1, and 4.

## Models

- We will use the Vision Transformer (vit_base_patch16_224.augreg_in21k).
- We will use the ResNet-50 (resnetv2_50x1_bit.goog_in21k).

Both models are pretrained on ImageNet-21k.

## Datasets

- **CIFAR-10**: We will evaluate using 10% of the dataset.
- **CIFAR-100**: We will evaluate using 10% and 100% of the dataset.

## Experiment Setup

For each combination of PEFT method, model, dataset, and epsilon value, we will record:

- Optimized batch size
- Optimized epochs
- Optimized learning rate
- Optimized max gradient norm
- Accuracy

## Results

### 01-film-vs-lora-evaluation

#### CIFAR10 (10% Subset, 20 Trials)

##### resnetv2_50x1_bit.goog_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 70 | 0.002912 | 1.24 | 0.82 |
| lora | 0.25 | 4500 | 67 | 0.002658 | 1.03 | 0.79 |
| film | 1.0 | 4500 | 53 | 0.003320 | 1.61 | 0.92 |
| lora | 1.0 | 4500 | 73 | 0.002729 | 1.34 | 0.90 |
| film | 4.0 | 512 | 114 | 0.001560 | 0.74 | 0.95 |
| lora | 4.0 | 4500 | 75 | 0.003496 | 0.99 | 0.94 |

#### CIFAR10 (10% Subset, 20 Trials)

##### vit_base_patch16_224.augreg_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 84 | 0.003055 | 1.62 | 0.88 |
| lora | 0.25 | 4500 | 67 | 0.002806 | 1.16 | 0.88 |
| film | 1.0 | 4500 | 66 | 0.002358 | 0.58 | 0.96 |
| lora | 1.0 | 4500 | 69 | 0.002092 | 1.70 | 0.96 |
| film | 4.0 | 512 | 78 | 0.001434 | 0.20 | 0.98 |
| lora | 4.0 | 4500 | 77 | 0.003036 | 2.13 | 0.99 |

#### CIFAR100 (10% Subset, 20 Trials)

##### resnetv2_50x1_bit.goog_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 99 | 0.003687 | 2.33 | 0.04 |
| lora | 0.25 | 4500 | 131 | 0.002147 | 0.60 | 0.05 |
| film | 1.0 | 4500 | 64 | 0.003116 | 1.69 | 0.29 |
| lora | 1.0 | 4500 | 77 | 0.003629 | 1.83 | 0.29 |
| film | 4.0 | 4500 | 76 | 0.004504 | 1.66 | 0.74 |
| lora | 4.0 | 4500 | 73 | 0.002554 | 1.43 | 0.70 |

#### CIFAR100 (100% Subset, 20 Trials)

##### resnetv2_50x1_bit.goog_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 45000 | 79 | 0.002686 | 1.51 | 0.60 |
| lora | 0.25 | 45000 | 75 | 0.002624 | 0.96 | 0.56 |
| film | 1.0 | 45000 | 79 | 0.002674 | 1.41 | 0.78 |
| lora | 1.0 | 45000 | 65 | 0.002699 | 1.10 | 0.73 |
| film | 4.0 | 45000 | 51 | 0.002974 | 1.73 | 0.83 |
| lora | 4.0 | 45000 | 70 | 0.003377 | 1.35 | 0.80 |

#### CIFAR100 (10% Subset, 20 Trials)

##### vit_base_patch16_224.augreg_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 72 | 0.004300 | 2.15 | 0.07 |
| lora | 0.25 | 256 | 123 | 0.000562 | 6.97 | 0.04 |
| film | 1.0 | 512 | 91 | 0.003061 | 2.30 | 0.56 |
| lora | 1.0 | 4500 | 77 | 0.003654 | 1.25 | 0.38 |
| film | 4.0 | 512 | 84 | 0.002352 | 0.70 | 0.86 |
| lora | 4.0 | 4500 | 67 | 0.002799 | 1.16 | 0.86 |

#### CIFAR100 (100% Subset, 20 Trials)

##### vit_base_patch16_224.augreg_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 2048 | 71 | 0.002014 | 4.83 | 0.75 |
| lora | 0.25 | 45000 | 93 | 0.003083 | 1.92 | 0.76 |
| film | 1.0 | 2048 | 83 | 0.000634 | 4.52 | 0.87 |
| lora | 1.0 | 45000 | 63 | 0.002848 | 2.29 | 0.89 |
| film | 4.0 | 45000 | 78 | 0.003256 | 3.63 | 0.91 |
| lora | 4.0 | 45000 | 79 | 0.002337 | 1.38 | 0.92 |

---

### 01-film-vs-lora-evaluation - Extension 50 Trials, 10% Subset

#### CIFAR10 (10% Subset, 50 Trials)

##### resnetv2_50x1_bit.goog_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 88 | 0.002956 | 1.24 | 0.83 |
| lora | 0.25 | 4500 | 69 | 0.002954 | 0.81 | 0.80 |
| film | 1.0 | 4500 | 52 | 0.003443 | 1.61 | 0.92 |
| lora | 1.0 | 4500 | 74 | 0.002967 | 1.24 | 0.92 |
| film | 4.0 | 512 | 109 | 0.001789 | 1.36 | 0.95 |
| lora | 4.0 | 4500 | 77 | 0.003449 | 0.96 | 0.95 |

#### CIFAR10 (10% Subset, 50 Trials)

##### vit_base_patch16_224.augreg_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 84 | 0.003564 | 1.62 | 0.89 |
| lora | 0.25 | 4500 | 64 | 0.002730 | 1.37 | 0.89 |
| film | 1.0 | 4500 | 77 | 0.002739 | 0.52 | 0.96 |
| lora | 1.0 | 4500 | 69 | 0.002092 | 1.70 | 0.96 |
| film | 4.0 | 4096 | 150 | 0.003432 | 0.20 | 0.99 |
| lora | 4.0 | 4500 | 73 | 0.002497 | 1.98 | 0.99 |

#### CIFAR100 (10% Subset, 50 Trials)

##### resnetv2_50x1_bit.goog_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 83 | 0.002598 | 2.37 | 0.05 |
| lora | 0.25 | 4500 | 125 | 0.002917 | 1.68 | 0.06 |
| film | 1.0 | 4500 | 57 | 0.003803 | 1.81 | 0.32 |
| lora | 1.0 | 4500 | 77 | 0.003629 | 1.83 | 0.29 |
| film | 4.0 | 4500 | 76 | 0.004504 | 1.66 | 0.74 |
| lora | 4.0 | 4500 | 73 | 0.002554 | 1.43 | 0.70 |

#### CIFAR100 (10% Subset, 50 Trials)

##### vit_base_patch16_224.augreg_in21k

| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 94 | 0.002715 | 2.98 | 0.08 |
| lora | 0.25 | 256 | 91 | 0.000440 | 6.55 | 0.06 |
| film | 1.0 | 512 | 90 | 0.003173 | 3.09 | 0.58 |
| lora | 1.0 | 4500 | 75 | 0.003480 | 1.33 | 0.42 |
| film | 4.0 | 512 | 122 | 0.001558 | 2.40 | 0.86 |
| lora | 4.0 | 4500 | 72 | 0.002988 | 1.56 | 0.87 |

#### Plots

##### Accuracy 20 vs 50 Trials

---

### 01-film-vs-lora-evaluation - Extension Full batch, 10% Subset

#### CIFAR10 (10% Subset, 20 Trials, Full batch)

##### resnetv2_50x1_bit.goog_in21k

| PEFT Method | Epsilon | Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 77 | 0.002937 | 0.20 | 0.81 |
| lora | 0.25 | 4500 | 111 | 0.005391 | 0.40 | 0.78 |
| film | 1.0 | 4500 | 54 | 0.003284 | 5.20 | 0.91 |
| lora | 1.0 | 4500 | 54 | 0.003178 | 1.19 | 0.91 |
| film | 4.0 | 4500 | 200 | 0.001202 | 6.52 | 0.94 |
| lora | 4.0 | 4500 | 71 | 0.003598 | 6.42 | 0.94 |

#### CIFAR10 (10% Subset, 20 Trials, Full batch)

##### vit_base_patch16_224.augreg_in21k

| PEFT Method | Epsilon | Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 45 | 0.005225 | 2.01 | 0.86 |
| lora | 0.25 | 4500 | 106 | 0.001901 | 0.66 | 0.87 |
| film | 1.0 | 4500 | 50 | 0.003248 | 0.42 | 0.95 |
| lora | 1.0 | 4500 | 137 | 0.001712 | 0.20 | 0.96 |
| film | 4.0 | 4500 | 33 | 0.001743 | 0.66 | 0.98 |
| lora | 4.0 | 4500 | 102 | 0.001811 | 0.20 | 0.99 |

#### CIFAR100 (10% Subset, 20 Trials, Full batch)

##### resnetv2_50x1_bit.goog_in21k

| PEFT Method | Epsilon | Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 75 | 0.003133 | 2.39 | 0.04 |
| lora | 0.25 | 4500 | 12 | 0.015742 | 6.09 | 0.05 |
| film | 1.0 | 4500 | 58 | 0.004434 | 4.41 | 0.31 |
| lora | 1.0 | 4500 | 102 | 0.003239 | 3.18 | 0.30 |
| film | 4.0 | 4500 | 81 | 0.004549 | 0.20 | 0.74 |
| lora | 4.0 | 4500 | 102 | 0.003285 | 0.25 | 0.68 |

#### CIFAR100 (10% Subset, 20 Trials, Full batch)

##### vit_base_patch16_224.augreg_in21k

| PEFT Method | Epsilon | Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| film | 0.25 | 4500 | 92 | 0.005141 | 2.16 | 0.05 |
| lora | 0.25 | 4500 | 200 | 0.000611 | 8.74 | 0.03 |
| film | 1.0 | 4500 | 200 | 0.005743 | 3.61 | 0.55 |
| lora | 1.0 | 4500 | 92 | 0.005141 | 2.16 | 0.12 |
| film | 4.0 | 4500 | 56 | 0.005771 | 5.07 | 0.83 |
| lora | 4.0 | 4500 | 70 | 0.003248 | 1.74 | 0.86 |
