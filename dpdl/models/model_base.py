import logging
import os

from collections import OrderedDict
from collections.abc import Mapping
from typing import Dict, Tuple, Any, Optional

import torch
import torchmetrics

log = logging.getLogger(__name__)


class ModelBase(torch.nn.Module):
    def __init__(
        self,
        model_instance: torch.nn.Module = None,
        num_classes: int = 10,
        use_feature_cache: bool = False,
        criterion: torch.nn.Module = None,
        metrics: Optional[Dict[str, Any]] = None
    ):

        super().__init__()

        self.model = model_instance
        self.num_classes = num_classes
        self.use_feature_cache = use_feature_cache

        if not criterion:
            raise ValueError('Criterion not passed to ModelBase.')

        self._criterion = criterion.cuda()

        if metrics is not None:
            self.train_metrics = metrics['train_metrics']
            self.valid_metrics = metrics['valid_metrics']
            self.test_metrics = metrics['test_metrics']

    @property
    def config(self):
        return self.model.config

    @property
    def prepare_inputs_for_generation(self):
        """Expose the underlying model's method."""

        # NB: This is used when calling `model.generate()`
        return self.model.prepare_inputs_for_generation

    def set_metrics(self, metrics):
        self.train_metrics = metrics["train_metrics"]
        self.valid_metrics = metrics["valid_metrics"]
        self.test_metrics = metrics["test_metrics"]

    def forward(self, *args, **kwargs):

        # If PEFT calls with keyword arguments, convert them to a dict and pass as x
        if kwargs and not args:
            if isinstance(kwargs.get('input_ids'), Mapping):
                x = kwargs['input_ids']
            else:
                x = kwargs
        elif args:
            x = args[0]
        else:
            x = None

        if self.use_feature_cache:
            return self.model.forward_head(x)
        else:
            return self.model(x)

    def forward_head(self, x):
        return self.model.forward_head(x)

    def forward_features(self, x):
        return self.model.forward_features(x)

    def criterion(self, logits, targets):
        return self._criterion(logits, targets)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def show_layers(self):
        log.info("Layers:")

        for n, m in self.model.named_modules():
            log.info(f"{n}, {type(m)}")

    def zero_head_weights(self):
        classifier = self.model.get_classifier()
        torch.nn.init.zeros_(classifier.weight)
        if classifier.bias is not None:
            torch.nn.init.zeros_(classifier.bias)

    def get_classifier(self):
        return self.model.get_classifier()

    def get_body(self):
        return torch.nn.Sequential(*list(self.model.children())[:-1])

    def save_model(self, fpath):
        # Extract the directory from the path
        directory = os.path.dirname(fpath)

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self.model.save_model(fpath)

    def load_model(
        self,
        fpath: str,
        *,
        strict: bool = True,
        map_location: str = 'cuda',
        allow_partial: bool = False,
        remap: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Load weights into self.model.

        Args:
            fpath: Path to checkpoint file on disk.
            strict: Passed to load_state_dict.
            map_location: torch.load map_location (e.g. 'cpu' or 'cuda').
            allow_partial: If True, drop keys not present in target before loading.
            remap: Optional explicit prefix remap {old_prefix: new_prefix}.
        """
        if not fpath:
            raise ValueError('load_model: fpath is required')

        if not os.path.isfile(fpath):
            raise FileNotFoundError(f'Checkpoint not found: {fpath}')

        ckpt = torch.load(fpath, map_location=map_location)

        # extract a plausible state_dict
        state = None
        if isinstance(ckpt, dict):
            for key in ('state_dict', 'model_state_dict', 'model', 'net', 'weights'):
                v = ckpt.get(key)
                if isinstance(v, dict):
                    state = v
                    break

            if state is None:
                # maybe it's already a raw state_dict
                state = {k: v for k, v in ckpt.items() if torch.is_tensor(v) or hasattr(v, 'shape')}
                if not state:
                    raise ValueError(f'No state_dict found in checkpoint: {fpath}')
        else:
            raise TypeError(f'Unexpected checkpoint type: {type(ckpt)}')

        def strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
            p = f'{prefix}.'
            return {(k[len(p) :] if k.startswith(p) else k): v for k, v in sd.items()}

        def add_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
            return {f'{prefix}.{k}': v for k, v in sd.items()}

        # always remove DDP wrapper if present
        state = strip_prefix(state, 'module')

        # optional explicit remap first (highest precedence)
        if remap:
            for old, new in remap.items():
                state = {
                    (k.replace(old + '.', new + '.') if k.startswith(old + '.') else k): v
                    for k, v in state.items()
                }

        target = self.model  # load into the the wrapper model
        tgt_keys = set(target.state_dict().keys())

        # build candidates and pick the one with max overlap
        roots = ['model', 'net', 'backbone', 'encoder']
        candidates: Dict[str, Dict[str, torch.Tensor]] = {'identity': state}

        for r in roots:
            candidates[f'add:{r}'] = add_prefix(state, r)

        for r in roots:
            candidates[f'remove:{r}'] = strip_prefix(state, r)

        def overlap(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
            ks = set(sd.keys())
            return (len(ks & tgt_keys), len(ks))

        best_name, best_sd, best_match = None, None, (-1, 1)
        for name, cand in candidates.items():
            m = overlap(cand)
            if m[0] > best_match[0] or (m[0] == best_match[0] and m[1] < best_match[1]):
                best_name, best_sd, best_match = name, cand, m

        state = best_sd
        matched, total = best_match
        if matched == 0:
            log.warning(
                'load_model: no key overlap with target; will rely on strict=%s/allow_partial=%s',
                strict,
                allow_partial,
            )

        # load the weights
        result = target.load_state_dict(OrderedDict(state), strict=strict)

        # diagnostics
        missing = getattr(result, 'missing_keys', [])
        unexpected = getattr(result, 'unexpected_keys', [])
        loaded_keys = matched if not allow_partial else len(state)
        pct = (loaded_keys / max(1, len(tgt_keys))) * 100.0

        if missing:
            log.warning(f'load_model: missing keys ({len(missing)}): {missing[:20]}')
        if unexpected:
            log.warning(f'load_model: unexpected keys ({len(unexpected)}): {unexpected[:20]}')

        log.info(
            f'Loaded weights from {fpath} using candidate={best_name} '
            f'(match {matched}/{len(tgt_keys)}, ~{pct:.1f}% of target).'
        )
