# Models and PEFT

In this document, we summarize how DPDL handles models, covering three model sources: [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models), [HuggingFace](https://huggingface.co/models), and custom.
Further we describe how to use parameter‑efficient fine‑tuning ([PEFT](https://github.com/huggingface/peft)) with the models.
We aim keeep the document brief and point to code for further details.

## Overview

Model creation happens in [ModelFactory](../dpdl/models/model_factory.py).
The main selection logic is:
- If `--llm` is set, load a HuggingFace model via [HuggingfaceLanguageModel](../dpdl/models/hugging_face_models.py).
- Otherwise, by default we use `timm` models, unless the model name matches a custom model or one of the provided ones (e.g., `wide_res_net-<depth>-<width>` for [WideResNet](https://arxiv.org/abs/1605.07146) or `koskela_net`).

After initilization, we wrap the models in [ModelBase](../dpdl/models/model_base.py) what provides a common interface (e.g. loss, metrics, save/load, etc) for the underlying models.

## Timm models (vision)

When `--llm` is **not** set and the model name does not match a custom model, [ModelFactory](../dpdl/models/model_factory.py) calls `timm.create_model()` to instatiate the requested model.
The factory gets the parameters from [ConfigurationManager](../dpdl/callbacks/configurationmanager.py) and creates the model as requested.
The model requested model is specified with `--model-name` CLI switch and loading pretrained weights is controlled by `--pretrained`/`--no-pretrained` switch.

## HuggingFace models (language)

When `--llm` is enabled, DPDL uses [HuggingfaceLanguageModel](../dpdl/models/hugging_face_models.py) to load HF models via the [Transfomers API](https://huggingface.co/docs/transformers/index).

If you need model‑specific behaviors (e.g., special tokenization), consult the upstream model docs and extend `HuggingfaceLanguageModel` as needed.

## Custom models

DPDL includes a small set of custom models, that have been implemented on a case-by-case needs:
- [WideResNet](https://arxiv.org/abs/1605.07146) via name pattern `wrn-<depth>-<width>`.
- KoskelaNet (used in [this paper](https://arxiv.org/pdf/1809.03832.pdf)) via name `koskela_net`.

Should you need a custom model, you can use it by either providing a path to the file or placing it in the models folder.
Note that when using the dpdl command line tool directly, only direct paths can be used due to Pythons handling of module paths.
Both methods require the model files to be in snake case and the model class to have the same name in capital camel case (e.g. dummy_net.py model file and DummyNet class).

In practice this means:
- Implement a new module while following the [ModelBase API](../dpdl/models/model_base.py).
    - The model has to at least expose the functions `forward`, `get_classifier` and `get_transforms`.
    - The model class name has to be the capital camel case version of the snake case file name (e.g. `dummy_net.py` model file and `DummyNet` class).
- provide the path as the model-name command line argument for the dpdl binary
- OR: provide the file name as the model-name command line argument if placed in the `models` folder and using `run.py`
    - In this case, model initialization parameters can be supplied by separation with hyphens, like so: `model_file-<param1>-<param2>`. Integers will be converted to the `int` type, all else will be passed as `str`.

## PEFT (parameter‑efficient fine‑tuning)

We provide an [implementation](../dpdl/peft.py) for some PEFT models.
PEFT is configured via `--peft MODE`
- `lora`: [Low‑rank adapters](https://arxiv.org/abs/2106.09685) for supported timm/HF model families.
- `film`: [FiLM](https://arxiv.org/abs/1709.07871) parameterization (e.g., train only norm scale/bias plus the head).
- `head-only`: Train only the (classifier) head.

Please note that LoRA/FiLM require model‑specific patterns and unsupported model names will raise an error until a new config is added in [peft.py](../dpdl/peft.py).

Additionally, the classifier (head) can be initialized to zero using the `--zero-head` option, following [prior work](https://arxiv.org/abs/2302.01190).

### Why PEFT can be useful for DP

In DP training, larger numbers of trainable parameters tend to increase average per‑step gradient magnitudes, which in turn increases the scale of injected noise for a fixed clipping bound.

This is one reason PEFT can be attractive under tight privacy budgets: it keeps optimization focused on a smaller parameter subset while reducing the effective scale of DP noise.

## Summary and a word about multi-modal models

In practice, the main distinction is:
- **Vision**: timm models + image transforms.
- **Language**: HuggingFace models + tokenizer transforms.

A natural future extension is multimodal models, which likely would require a new model wrapper and adapter logic to handle mixed inputs.

