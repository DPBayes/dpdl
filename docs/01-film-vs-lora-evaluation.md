# PEFT Methods in Differentially Private Deep Learning

## Objective

We aim to evaluate the effectiveness of different Parameter Efficient Fine-tuning (PEFT) methods—FiLM and LoRA—in the context of differentially private deep learning to determine the most suitable method for our future experiments.

## Methodology

- **PEFT Methods**: We will compare two PEFT methods—FiLM and LoRA.
- **Hyperparameter Optimization**: We will use Bayesian optimization to fine-tune all hyperparameters (epochs, learning_rate, batch_size, max_grad_norm) for each PEFT method.
- **Epsilon Values**: We will conduct experiments for each PEFT method with differential privacy guarantees at epsilon values of 1.0 and 8.0.

## Models

- We will use the Vision Transformer (vit_base_patch16_224.augreg_in21k).
- We will use the ResNet-50 (resnetv2_50x1_bit.goog_in21k).

Both models are pretrained on ImageNet-21k.

## Datasets

- **CIFAR-10**: We will evaluate using 10% of the dataset.
- **CIFAR-100**: We will evaluate using 10% and 100% of the dataset.

## Experiment Setup

For each combination of PEFT method, model, dataset, and epsilon value, we will record:

- Optimized epochs
- Optimized learning rate
- Optimized max gradient norm
- Accuracy

## Results

### CIFAR-10 (10% subset)

#### Vision Transformer (vit_base_patch16_224.augreg_in21k)

| PEFT Method | Epsilon | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|------------------|-------------------------|-----------------------------|----------|
| FiLM        | 1.0     |                  |                         |                             |          |
| LoRA        | 1.0     |                  |                         |                             |          |
| FiLM        | 8.0     |                  |                         |                             |          |
| LoRA        | 8.0     |                  |                         |                             |          |

#### ResNet-50 (resnetv2_50x1_bit.goog_in21k)

| PEFT Method | Epsilon | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|---------|------------------|-------------------------|-----------------------------|----------|
| FiLM        | 1.0     |                  |                         |                             |          |
| LoRA        | 1.0     |                  |                         |                             |          |
| FiLM        | 8.0     |                  |                         |                             |          |
| LoRA        | 8.0     |                  |                         |                             |          |

### CIFAR-100 (10% and 100% subsets)

#### Vision Transformer (vit_base_patch16_224.augreg_in21k)

| PEFT Method | Dataset Subset | Epsilon | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|----------------|---------|------------------|-------------------------|-----------------------------|----------|
| FiLM        | 10%            | 1.0     |                  |                         |                             |          |
| LoRA        | 10%            | 1.0     |                  |                         |                             |          |
| FiLM        | 100%           | 1.0     |                  |                         |                             |          |
| LoRA        | 100%           | 1.0     |                  |                         |                             |          |
| FiLM        | 10%            | 8.0     |                  |                         |                             |          |
| LoRA        | 10%            | 8.0     |                  |                         |                             |          |
| FiLM        | 100%           | 8.0     |                  |                         |                             |          |
| LoRA        | 100%           | 8.0     |                  |                         |                             |          |

#### ResNet-50 (resnetv2_50x1_bit.goog_in21k)

| PEFT Method | Dataset Subset | Epsilon | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |
|-------------|----------------|---------|------------------|-------------------------|-----------------------------|----------|
| FiLM        | 10%            | 1.0     |                  |                         |                             |          |
| LoRA        | 10%            | 1.0     |                  |                         |                             |          |
| FiLM        | 100%           | 1.0     |                  |                         |                             |          |
| LoRA        | 100%           | 1.0     |                  |                         |                             |          |
| FiLM        | 10%            | 8.0     |                  |                         |                             |          |
| LoRA        | 10%            | 8.0     |                  |                         |                             |          |
| FiLM        | 100%           | 8.0     |                  |                         |                             |          |
| LoRA        | 100%           | 8.0     |                  |                         |                             |          |
