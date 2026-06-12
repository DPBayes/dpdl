import os
import logging

from .hugging_face_models import HuggingfaceLanguageModel

from dpdl.configurationmanager import Configuration

log = logging.getLogger(__name__)



class LLMBuilder:

    @staticmethod
    def matches(configuration):
        return configuration.task in ("CausalLM", "InstructLM", "SequenceClassification")

    @staticmethod
    def get_model(
        configuration: Configuration,
        output_dim: int | None,
        checkpoints_dir_latest: str | None = None,
    ):


        model_instance = HuggingfaceLanguageModel(
            configuration.model_name,
            configuration.load_in_4bit,
            num_labels=output_dim,
            peft=configuration.peft,
            checkpoint_dir=checkpoints_dir_latest,
            task=configuration.task,
        )

        transforms = model_instance.get_transforms()

        if output_dim is None:
            try:
                output_dim = int(model_instance.model.num_classes) \
                    if configuration.task == "SequenceClassification" \
                    else int(model_instance.config.vocab_size)
            except AttributeError:
                raise ValueError('Output dimension not given and unable to infer it.')

        return model_instance, transforms, output_dim