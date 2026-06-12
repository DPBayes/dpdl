import timm
import logging

from dpdl.configurationmanager import Configuration

log = logging.getLogger(__name__)


class TimmBuilder:

    @staticmethod
    def get_model(
            configuration: Configuration,
            output_dim: int | None
    ):
        model_instance = timm.create_model(
            configuration.model_name,
            pretrained=configuration.pretrained,
            num_classes=output_dim,
        )

        # Resolve data config and create transforms
        model_config = timm.data.resolve_data_config({}, model=model_instance)
        transforms = timm.data.transforms_factory.create_transform(**model_config)

        if output_dim is None:
            output_dim = model_instance.num_classes

        return model_instance, transforms, output_dim


