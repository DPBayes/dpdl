import logging

from .wide_resnet import WideResNet
from .koskela_model import KoskelaNet

from dpdl.configurationmanager import Configuration

log = logging.getLogger(__name__)


class CustomBuilder:

    @staticmethod
    def matches(configuration):
        return configuration.model_name.startswith('wrn-') or configuration.model_name == 'koskela-net'

    @staticmethod
    def get_model(
            configuration: Configuration,
            output_dim: int | None
    ):
        if 'wrn' in configuration.model_name:
            parts = configuration.model_name.split('-')
            depth, width = int(parts[1]), int(parts[2])
            model_instance = WideResNet(depth=depth, width=width, num_classes=output_dim)
        elif configuration.model_name == 'koskela-net':
            model_instance = KoskelaNet(num_classes=output_dim)
        else:
            raise ValueError(f"Custom model class could not be determined from name: {configuration.model_name}")

        transforms = model_instance.get_transforms()

        return model_instance, transforms, output_dim