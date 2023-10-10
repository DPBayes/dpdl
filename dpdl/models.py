import logging
import opacus
import timm
import torch
import torchmetrics

from peft import get_peft_model, LoraConfig

from .configurationmanager import Configuration, Hyperparameters

log = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def get_model(configuration: Configuration, hyperparams: Hyperparameters):
        if configuration.lora:
            model = ModelFactory._get_lora_model(configuration, hyperparams)
        else:
            model = ModelFactory._get_basic_model(configuration, hyperparams)

        return model

    def _get_basic_model(configuration: Configuration, hyperparams: Hyperparameters):
        model = ImageClassificationModel(
            model_name=configuration.model_name,
            num_classes=configuration.num_classes,
            fix_model=configuration.modulevalidator_fix,
        )

        return model

    def _get_lora_model(configuration: Configuration, hyperparams: Hyperparameters):
        model = ModelFactory._get_basic_model(configuration, hyperparams)

        lora_config = ModelFactory._get_lora_config(configuration, hyperparams)
        peft_model = get_peft_model(model, lora_config)

        trainable_params, all_params = peft_model.get_nb_trainable_parameters()
        log.info(f'LoRA setup done - trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {100 * trainable_params / all_params}')

        return peft_model

    def _get_lora_config(configuration: Configuration, hyperparams: Hyperparameters):
        if configuration.model_name.startswith('vit_base_patch16_224'):
            lora_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=8,
                bias='none',
                target_modules=r".*\.patch_embed.proj|.*\.attn\.qkv|.*\.attn\.proj|.*\.mlp\.fc\d",
                modules_to_save=['head'],
            )

            return lora_config

        if configuration.model_name.startswith('resnetv2_50x1_bit'):
            lora_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=8,
                bias='none',
                target_modules=r"stem\.conv|.*\.conv\d",
                modules_to_save=['head.fc'],
            )

            return lora_config

        raise RuntimeError(f'No known LoRA configuration for model: {configuration.model_name}')

class TimmModel(torch.nn.Module):
    def __init__(
            self,
            *,
            model_name: str = 'resnet18',
            pretrained: bool = True,
            fix_model: bool = False,
            **kwargs,
        ):

        super().__init__(**kwargs)

        self.model_name = model_name
        self.pretrained = pretrained
        self.fix_model = fix_model

        # no default metrics
        self.train_metrics = torchmetrics.MetricCollection([])
        self.valid_metrics = torchmetrics.MetricCollection([])

    def forward(self, x):
        return self.model(x)

    def criterion(self, logits, y):
        raise NotImplementedError('Criterion not implemented for class: {self.__class__.__name__}')

    def show_layers(self):
        log.info('Layers:')

        for n, m in self.model.named_modules():
            log.info(f'{n}, {type(m)}')

class ImageClassificationModel(TimmModel):
    def __init__(
            self,
            *,
            num_classes: int = 10,
            **kwargs,
        ):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        self.model = timm.create_model(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=self.pretrained
        )

        if not opacus.validators.ModuleValidator.is_valid(self.model):
            if self.fix_model:
                self.model = opacus.validators.ModuleValidator.fix(self.model)
            else:
                raise RuntimeError("Model contains layers that are note compatible with DP-SGD. "
                                   "Use --modulevalidator-fix (with caution!) to automatically fix the model.")

        self._criterion = torch.nn.CrossEntropyLoss().cuda()

        # let's track the accuracy
        self.train_metrics = torchmetrics.MetricCollection([
            torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes),
        ])

        self.valid_metrics = torchmetrics.MetricCollection([
            torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes),
        ])

    def criterion(self, logits, y):
        return self._criterion(logits, y)
