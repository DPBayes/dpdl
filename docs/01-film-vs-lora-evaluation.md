# PEFT Methods in Differentially Private Deep Learning

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

### CIFAR-10 (10% subset)

#### Vision Transformer (vit_base_patch16_224.augreg_in21k)

| PEFT Method | Epsilon | Optimized Batch Size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| FiLM        | 0.25    |                      |                  |                         |                             |          |
| LoRA        | 0.25    |                      |                  |                         |                             |          |
| FiLM        | 1       |                      |                  |                         |                             |          |
| LoRA        | 1       |                      |                  |                         |                             |          |
| FiLM        | 4       |                      |                  |                         |                             |          |
| LoRA        | 4       |                      |                  |                         |                             |          |

#### ResNet-50 (resnetv2_50x1_bit.goog_in21k)

| PEFT Method | Epsilon | Optimized Batch Size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| FiLM        | 0.25    |                      |                  |                         |                             |          |
| LoRA        | 0.25    |                      |                  |                         |                             |          |
| FiLM        | 1       |                      |                  |                         |                             |          |
| LoRA        | 1       |                      |                  |                         |                             |          |
| FiLM        | 4       |                      |                  |                         |                             |          |
| LoRA        | 4       |                      |                  |                         |                             |          |

### CIFAR-100 (10% and 100% subsets)

#### Vision Transformer (vit_base_patch16_224.augreg_in21k)

| PEFT Method | Dataset Subset | Epsilon | Optimized Batch Size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|----------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| FiLM        | 10%            | 0.25    |                      |                  |                         |                             |          |
| LoRA        | 10%            | 0.25    |                      |                  |                         |                             |          |
| FiLM        | 10%            | 1       |                      |                  |                         |                             |          |
| LoRA        | 10%            | 1       |                      |                  |                         |                             |          |
| FiLM        | 10%            | 4       |                      |                  |                         |                             |          |
| LoRA        | 10%            | 4       |                      |                  |                         |                             |          |
| FiLM        | 100%           | 0.25    |                      |                  |                         |                             |          |
| LoRA        | 100%           | 0.25    |                      |                  |                         |                             |          |
| FiLM        | 100%           | 1       |                      |                  |                         |                             |          |
| LoRA        | 100%           | 1       |                      |                  |                         |                             |          |
| FiLM        | 100%           | 4       |                      |                  |                         |                             |          |
| LoRA        | 100%           | 4       |                      |                  |                         |                             |          |

#### ResNet-50 (resnetv2_50x1_bit.goog_in21k)

| PEFT Method | Dataset Subset | Epsilon | Optimized Batch Size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|----------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|
| FiLM        | 10%            | 0.25    |                      |                  |                         |                             |          |
| LoRA        | 10%            | 0.25    |                      |                  |                         |                             |          |
| FiLM        | 10%            | 1       |                      |                  |                         |                             |          |
| LoRA        | 10%            | 1       |                      |                  |                         |                             |          |
| FiLM        | 10%            | 4       |                      |                  |                         |                             |          |
| LoRA        | 10%            | 4       |                      |                  |                         |                             |          |
| FiLM        | 100%           | 0.25    |                      |                  |                         |                             |          |
| LoRA        | 100%           | 0.25    |                      |                  |                         |                             |          |
| FiLM        | 100%           | 1       |                      |                  |                         |                             |          |
| LoRA        | 100%           | 1       |                      |                  |                         |                             |          |
| FiLM        | 100%           | 4       |                      |                  |                         |                             |          |
| LoRA        | 100%           | 4       |                      |                  |                         |                             |          |
