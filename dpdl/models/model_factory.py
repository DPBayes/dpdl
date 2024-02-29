import logging
import timm

from .model_base import ModelBase
from .timm_model import TimmModel
from .wide_resnet import WideResNet
from .koskela_model import KoskelaNet

from dpdl.configurationmanager import Configuration, Hyperparameters
from dpdl.peft import PeftFactory

log = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def get_model(configuration, hyperparams=None):
        """
        Create a model instance based on the configuration, with support for PEFT and zeroing head weights.

        Parameters:
        - configuration: Configuration object containing model specs.
        - hyperparams: Optional hyperparameters, not directly used here.

        Returns:
        - A tuple of (ModelBase instance, Data Transforms).
        """

        transforms = {}  # No default transforms
        model_instance = None

        if configuration.model_name.startswith('wrn-'):
            # Parse depth and width from model_name, e.g., 'wrn-16-4'
            parts = configuration.model_name.split('-')
            depth, width = int(parts[1]), int(parts[2])
            model_instance = WideResNet(depth=depth, width=width, num_classes=configuration.num_classes)
            transforms = model_instance.get_transforms()
        elif configuration.model_name == 'koskela-net':
            model_instance = KoskelaNet()
            transforms = model_instance.get_transforms()
        else:
            # Default to using TimmModel
            model_instance = TimmModel(
                model_name=configuration.model_name,
                num_classes=configuration.num_classes,
                pretrained=configuration.pretrained,
            )

            # Resolve data config and create transforms
            model_config = timm.data.resolve_data_config({}, model=model_instance.model)
            transforms = timm.data.transforms_factory.create_transform(**model_config)

        # Wrap the instantiated model with ModelBase
        model = ModelBase(model_instance=model_instance, num_classes=configuration.num_classes)

        # zero the head weights?
        if configuration.zero_head:
            model.zero_head_weights()

        # should we do Parameter Efficient Fine-Tuning (PEFT)?
        if configuration.peft:
            model = PeftFactory.get_peft_model(model, configuration)

        return model, transforms

