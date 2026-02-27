import torch

from dpdl.configurationmanager import Configuration, Hyperparameters
from dpdl.models.model_factory import ModelFactory
from dpdl.models.vgg_bnb_reference_model import VGGBnBReferenceModel


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def test_vgg_reference_default_param_count_matches_jax_reference() -> None:
    model = VGGBnBReferenceModel(
        num_classes=10,
        channels=[32, 64, 128],
        dense_size=128,
        activation='tanh',
        input_size=32,
    )
    assert _count_params(model) == 550_570


def test_vgg_reference_forward_shape() -> None:
    model = VGGBnBReferenceModel(num_classes=7)
    x = torch.randn(5, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (5, 7)


def test_model_factory_builds_vgg_reference_with_resolved_config() -> None:
    cfg = Configuration(
        command='train',
        privacy=False,
        model_name='vgg_bnb_reference',
    )
    hypers = Hyperparameters(
        privacy=False,
        epochs=1,
        batch_size=4,
        noise_multiplier=None,
        max_grad_norm=None,
        target_epsilon=None,
        noise_batch_ratio=None,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    model, transforms, num_classes = ModelFactory.get_model(
        configuration=cfg,
        hyperparams=hypers,
        num_classes=10,
        loss_fn=loss_fn,
    )

    assert isinstance(model.model, VGGBnBReferenceModel)
    assert model.model.channels == [32, 64, 128]
    assert model.model.dense_size == 128
    assert model.model.activation_name == 'tanh'
    assert model.model.input_size == 32
    assert transforms is not None
    assert num_classes == 10
