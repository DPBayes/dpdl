from __future__ import annotations

import logging
import math
import os
import time
import json
from collections.abc import Mapping

import opacus
import torch
from opacus import GradSampleModule
from opacus.accountants.analysis.bnb import (
    build_bnb_toeplitz_c_matrix_and_contract,
    resolve_bnb_calibration_kwargs,
)
from opacus.accountants.analysis.bsr import calibrate_bsr_z_std
from opacus.accountants.analysis.bsr import compute_bsr_kappa_from_coeffs
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel
from opacus.mechanism_contracts import (
    NoiseMechanismConfig,
    SamplingSemantics,
    resolve_accounting_mode_from_accountant,
)
from opacus.utils.batch_memory_manager import BatchMemoryManager

from peft import PeftModel

from .callbacks.callback_factory import CallbackFactory, CallbackHandler
from .bsr import generate_bsr_coeffs_from_sgd_workload
from .configurationmanager import Configuration, ConfigurationManager, Hyperparameters
from .datamodules import DataModule, DataModuleFactory
from .device import resolve_device
from .loss_factory import LossFactory
from .metrics_factory import MetricsFactory
from .models.model_base import ModelBase
from .models.model_factory import ModelFactory
from .optimizers import OptimizerFactory
from .utils import seed_everything, shift_and_flatten

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,

        # essentials
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        datamodule: DataModule,
        adapter: TaskAdapter,

        # generic params
        epochs: int = 10,
        total_steps: int = 0,
        validation_frequency: int = 1,
        seed: int = 0,
        physical_batch_size: int = 40,
        callback_handler: CallbackHandler | None = None,
        peft: str | None = None,
        task: str | None = None,
        device: torch.device | None = None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.datamodule = datamodule
        self.epochs = epochs
        self.total_steps = total_steps
        self.validation_frequency = validation_frequency
        self.seed = seed
        self.physical_batch_size = physical_batch_size
        self.peft = peft
        self.task = task
        self.device = device or torch.device('cuda')
        self.adapter = adapter
        self.adapter.device = self.device

        if not callback_handler:
            self.callback_handler = CallbackHandler()
        else:
            self.callback_handler = callback_handler

        if self.epochs and self.total_steps:
            raise ValueError('You should provide either "epochs" or "total_steps", not both.')

        self.setup()

    def setup(self):
        self.model = self.model.to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model)

    def fit(self):
        self.callback_handler.call('on_train_start', self)

        if self.total_steps:
            self._fit_total_steps()
        else:
            self._fit_epochs()

        self.callback_handler.call('on_train_end', self)

    def _fit_epochs(self):
        for epoch in range(self.epochs):
            self.fit_one_epoch(epoch)

            if self.validation_frequency and epoch % self.validation_frequency == 0:
                if torch.distributed.get_rank() == 0:
                    self.validate(epoch)

                # other ranks will wait for validation
                torch.distributed.barrier()

    def _fit_total_steps(self):
        step = 0
        virtual_epoch = 0
        steps_per_epoch = self._calculate_steps_per_epoch()

        # start the first virtual epoch
        self._handle_virtual_epoch_start(virtual_epoch)

        while step < self.total_steps:
            for batch_idx, batch in enumerate(self.datamodule.get_dataloader('train')):
                if step >= self.total_steps:
                    break

                self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)
                logical_batch_loss = self.fit_one_batch(batch_idx, batch)
                self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, logical_batch_loss)

                step += 1

                if step % steps_per_epoch == 0:
                    self._handle_virtual_epoch_end(virtual_epoch)
                    virtual_epoch += 1

                    if self.validation_frequency and virtual_epoch % self.validation_frequency == 0:
                        if torch.distributed.get_rank() == 0:
                            self.validate(virtual_epoch)

                        # other ranks will wait for validation
                        torch.distributed.barrier()

                    # are we finished?
                    if step >= self.total_steps:
                        break

                    # start the next virtual epoch
                    self._handle_virtual_epoch_start(virtual_epoch)

        last_step_in_epoch = step % steps_per_epoch
        if last_step_in_epoch != 0:
            self._handle_virtual_epoch_end(virtual_epoch)

        assert step == self.total_steps, f'Mismatch in total steps count: Expected {self.total_steps} total steps, but stepped {step} times!'

    def _handle_virtual_epoch_start(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

    def _handle_virtual_epoch_end(self, epoch):
        # compute the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()

        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        for batch_idx, batch in enumerate(self.datamodule.get_dataloader('train')):
            self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)

            logical_batch_loss = self.fit_one_batch(batch_idx, batch)

            self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, logical_batch_loss)

        # compute the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()

        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)

    def fit_one_batch(self, batch_idx, batch):
        X, y = batch
        X, y = self.adapter.move_to_device(X, y)

        # gradient accumulation. split the batch to sub batches that fit in the GPU memory.
        # then process the sub batches one at a time and call backward.
        # when all the sub batches have been processed we can finally step the optimizer.

        # the adapter handles the physical batches, as it's a different operation depending on the task.
        physical_batches = list(self.adapter.iterate_physical_batches((X, y), self.physical_batch_size))
        N = len(physical_batches)

        logical_batch_loss = 0.0

        # zero the grads as usually before doing anything
        self.optimizer.zero_grad()

        logical_batch_loss = 0
        for i, physical_batch in enumerate(physical_batches):
            self.callback_handler.call('on_train_physical_batch_start', self, i, physical_batch)

            forward_output = self.adapter.forward(self._unwrap_model(), physical_batch)
            loss = self.adapter.compute_loss(self._unwrap_model(), physical_batch, forward_output, normalize_by=N)
            self.adapter.update_metrics(self._unwrap_model(), physical_batch, forward_output)
            loss.backward()

            logical_batch_loss += loss.item()

            # notify the callbacks of a physical batch end
            self.callback_handler.call('on_train_physical_batch_end', self, i, physical_batch, loss.item())

        # after accumulating the gradients for all the sub batches we can finally update weights.
        self.optimizer.step()

        return logical_batch_loss

    def validate(self, epoch=None, enable_callbacks=True):
        return self._evaluate('validation', epoch, enable_callbacks)

    def test(self):
        return self._evaluate('test')

    def get_dataloader(self, name):
        return self.datamodule.get_dataloader(name)

    def get_datamodule(self):
        return self.datamodule

    def _evaluate(self, mode, epoch=None, enable_callbacks=True):
        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_epoch_start', self, epoch)

        self.model.eval()
        torch.set_grad_enabled(False)

        # record the loss separately, as we need to return it when
        # performing hyperparameter optimization
        evaluation_loss = 0

        if mode == 'validation':
            dataloader_name = 'valid'
            metrics_evaluator = self._unwrap_model().valid_metrics
        elif mode == 'test':
            dataloader_name = 'test'
            metrics_evaluator = self._unwrap_model().test_metrics
        elif mode == 'train':
            dataloader_name = 'train_eval'
            metrics_evaluator = self._unwrap_model().train_metrics
        else:
            raise ValueError(f'Unknown evaluation mode: "{mode}"')

        dataloader = self.datamodule.get_dataloader(dataloader_name)

        metrics_evaluator.reset()

        for batch_idx, batch in enumerate(dataloader):
            loss = self._evaluate_one_batch(mode, batch_idx, batch, enable_callbacks, metrics_evaluator)
            evaluation_loss += loss

        evaluation_loss /= len(dataloader)

        metrics = metrics_evaluator.compute()

        torch.set_grad_enabled(True)
        self.model.train()

        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_epoch_end', self, epoch, metrics)

        return evaluation_loss, metrics

    def _evaluate_one_batch(self, mode, batch_idx, batch, enable_callbacks, metrics_evaluator):
        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_batch_start', self, batch_idx, batch)

        X, y = batch
        X, y = self.adapter.move_to_device(X, y)

        forward_output = self.adapter.forward(self._unwrap_model(), (X, y))
        loss = self.adapter.compute_loss(self._unwrap_model(), (X, y), forward_output)
        self.adapter.update_metrics(
            self._unwrap_model(),
            (X, y),
            forward_output,
            metrics=metrics_evaluator,  # record into the provided evaluator
        )

        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_batch_end', self, batch_idx, batch, loss.item())

        return loss.item()

    def _unwrap_model(self):
        m = self.model

        # model can be wrapped inside many module, such as
        # DDP, Opacus' DPDDP or GradSampleModule, and HuggingFace's
        # PeftModule. Let's just unwrap the all the get to ModelBase
        while hasattr(m, 'module'):
            m = m.module

        return m  # ModelBase


    def _calculate_steps_per_epoch(self):
        N = len(self.datamodule.get_dataloader('train').dataset)
        B = self.datamodule.batch_size
        return math.ceil(N / B)

    def save_model(self, fpath, adapters_only=False):
        def unwrap_model_for_saving(m):
            # strip opacus and distributed models until we hit
            # either a ModelBase or HuggingFace's PeftModel
            while True:
                # Strip Opacus' GradSampleModule
                if isinstance(m, opacus.GradSampleModule):
                    m = m._module
                    continue

                # Strip Opacus' DP DPDDP
                if isinstance(m, opacus.distributed.DifferentiallyPrivateDistributedDataParallel):
                    m = m.module
                    continue

                # Strip standard DDP
                if isinstance(m, torch.nn.parallel.DistributedDataParallel):
                    m = m.module
                    continue

                # Stop when we if we found what we want
                if isinstance(m, (PeftModel, ModelBase)):
                    return m

            return m

        model = unwrap_model_for_saving(self.model)

        if isinstance(model, PeftModel):
            if adapters_only:
                # PeftModel knows to save the adapters only
                model.save_pretrained(fpath)

                log.info(f'Saved merged HF PEFT adapters to {fpath}')
            else:
                # Merge PEFT into model and save the whole model
                merged = model.merge_and_unload()

                log.info(f'GOT A NEW MODEL FROM MERGE_AND_UNLOAD: {merged}')
                # The `merge_and_unload` will incorporate the LoRA layers in
                # the model. Then it will return as ModelBase.
                merged.save_model(fpath)

                if torch.distributed.get_rank() == 0:
                    log.info(f'Saved merged HF PEFT model to {fpath}')

            return

        model.save_model(fpath)

    def _sample_impl(self):
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.datamodule.get_dataloader('sample')):
                X = batch
                X = self.adapter.move_to_device(X)

                is_mapping = isinstance(X, Mapping)  # covers dict and HF BatchEncoding
                # gradient accumulation. split the batch to sub batches that fit in the GPU memory.
                # then process the sub batches one at a time and call backward.
                # when all the sub batches have been processed we can finally step the optimizer.
                if is_mapping:
                    # split each tensor in the dict
                    X_split = {k: v.split(self.physical_batch_size, dim=0) for k, v in X.items()}
                else:
                    X_split = X.split(self.physical_batch_size, dim=0)

                N = len(X_split['input_ids'])

                for i in range(N):
                    if is_mapping:
                        X_splitted = {k: X_split[k][i] for k in X_split}
                    else:
                        X_splitted = X_split[i]

                    generated_ids = self._unwrap_model().generate(
                        X_splitted,
                        max_new_tokens=250,
                        temperature=0.5,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.datamodule.tokenizer.pad_token_id,
                        eos_token_id=self.datamodule.tokenizer.eos_token_id,
                    )

                    log.info('Sampled text decoded', self.datamodule.decode(generated_ids))

        self.model.train()


class DifferentiallyPrivateTrainer(Trainer):
    @staticmethod
    def _log_bsr_trace(
        *,
        stage: str,
        sampling_semantics: SamplingSemantics | None,
        noise_mechanism_config: NoiseMechanismConfig | None,
        has_target_privacy_params: bool,
        noise_multiplier_ref: float | None,
        correlated_denominator: float | None,
        mechanism_kwargs: dict,
    ) -> None:
        if noise_mechanism_config is None or noise_mechanism_config.mechanism != 'bsr':
            return

        state = noise_mechanism_config.mechanism_state
        coeffs = state.get('coeffs', [])
        metadata = sampling_semantics.privacy_metadata if sampling_semantics is not None else {}
        payload = {
            'stage': stage,
            'mechanism': noise_mechanism_config.mechanism,
            'accounting_mode': noise_mechanism_config.accounting_mode,
            'sampling_mode': sampling_semantics.sampling_mode if sampling_semantics is not None else None,
            'sampling_metadata': dict(metadata),
            'has_target_privacy_params': has_target_privacy_params,
            'noise_multiplier_ref': noise_multiplier_ref,
            'correlated_denominator': correlated_denominator,
            'mechanism_kwargs': dict(mechanism_kwargs),
            'mechanism_state': {
                'coeff_count': len(coeffs) if isinstance(coeffs, list) else None,
                'coeff_head': list(coeffs[:5]) if isinstance(coeffs, list) else None,
                'z_std': state.get('z_std'),
                'sensitivity_scale': state.get('sensitivity_scale'),
                'iterations_number': state.get('iterations_number'),
                'min_separation': state.get('min_separation'),
                'max_participations': state.get('max_participations'),
                'mf_sensitivity': state.get('mf_sensitivity'),
                'bands': state.get('bands'),
            },
        }
        log.info('BSR_TRACE %s', json.dumps(payload, sort_keys=True))

    def __init__(
        self,
        *,
        # privacy params
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        clipping_mode: str = 'flat',
        accountant: str = 'prv',
        poisson_sampling: bool = True,
        normalize_clipping: bool = False,
        noise_mechanism: str = 'gaussian',
        sampling_mode: str | None = None,
        bsr_coeffs: list[float] | None = None,
        bsr_z_std: float | None = None,
        bsr_bands: int | None = None,
        bsr_max_participations: int | None = None,
        bsr_min_separation: int | None = None,
        bsr_mf_sensitivity: float | None = None,
        bsr_iterations_number: int | None = None,
        bsr_alpha: float | None = None,
        bsr_beta: float | None = None,
        bnb_b: int | None = None,
        bnb_p: float | None = None,
        bnb_bands: int | None = None,
        bnb_num_samples: int | None = None,
        bnb_seed: int | None = None,
        secure_mode: bool = False,
        target_epsilon: float | None = None,
        target_delta: float | None = None,
        noise_batch_ratio: float | None = None,
        seed: int = 0,
        **kwargs,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.clipping_mode = clipping_mode
        self.accountant = accountant
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_batch_ratio = noise_batch_ratio
        self.seed = seed
        self.poisson_sampling = poisson_sampling
        self.normalize_clipping = normalize_clipping
        self.accountant = accountant
        self.noise_mechanism = noise_mechanism
        self.sampling_mode = sampling_mode
        self.bsr_coeffs = bsr_coeffs
        self.bsr_z_std = bsr_z_std
        self.bsr_bands = bsr_bands
        self.bsr_max_participations = bsr_max_participations
        self.bsr_min_separation = bsr_min_separation
        self.bsr_mf_sensitivity = bsr_mf_sensitivity
        self.bsr_iterations_number = bsr_iterations_number
        self.bsr_alpha = bsr_alpha
        self.bsr_beta = bsr_beta
        self.bnb_b = bnb_b
        self.bnb_p = bnb_p
        self.bnb_bands = bnb_bands
        self.bnb_num_samples = bnb_num_samples
        self.bnb_seed = bnb_seed

        # setup opacus privacy engine
        privacy_engine_args = {
            'accountant': accountant,
            'secure_mode': secure_mode,
        }

        self.privacy_engine = opacus.PrivacyEngine(**privacy_engine_args)

        super().__init__(seed=seed, **kwargs)

    def _has_target_privacy_params(self):
        if self.target_epsilon == -1:
            return False

        if self.target_epsilon is not None and self.target_epsilon < 0:
            raise ValueError('Parameter "target_epsilon" must be positive, or -1 for clip-only mode.')

        if (
            self.target_epsilon is None
            and self.noise_multiplier is None
            and self.noise_batch_ratio is None
        ):
            raise ValueError(
                'Privacy is enabled but no DP noise parameter was provided. '
                'Set one of "target_epsilon", "noise_multiplier", or "noise_batch_ratio".'
            )

        if not self.target_epsilon:
            return False

        if self.target_epsilon and not self.target_delta:
            raise ValueError('Parameter "target_epsilon" and "target_delta" not given.')

        if self.noise_batch_ratio and not self.target_delta:
            raise ValueError('Parameter "target_epsilon" and "target_delta" not given.')

        if all([self.target_epsilon, self.noise_batch_ratio]):
            raise ValueError('Parameters "target_epsilon" and "noise_batch_ratio" are exclusive.')

        if all([self.target_epsilon, self.noise_multiplier]):
            raise ValueError('Parameters "target_epsilon" and "noise_multiplier" are exlusive.')

        if all([self.noise_batch_ratio, self.noise_multiplier]):
            raise ValueError('Parameters "noise_batch_ratio" and "noise_multiplier" are exclusive.')

        if self.target_epsilon and not self.target_delta:
            raise ValueError('Parameter "target_epsilon" present, but "target_delta" is missing.')

        return True

    @staticmethod
    def _resolve_bsr_cyclic_sensitivity_scale(
        *,
        coeffs: list[float] | None,
        steps: int | None,
        iterations_number: int | None = None,
    ) -> float | None:
        if not coeffs:
            return None

        if steps is None:
            return None

        scale_steps = int(iterations_number) if iterations_number is not None else int(steps)
        if scale_steps < 1:
            raise ValueError('BSR cyclic sensitivity scale requires steps >= 1.')

        return float(
            compute_bsr_kappa_from_coeffs(
                coeffs=list(coeffs),
                steps=scale_steps,
            )
        )

    @staticmethod
    def _resolve_or_generate_bsr_coeffs(
        *,
        noise_mechanism: str,
        explicit_coeffs: list[float] | None,
        bsr_bands: int | None,
        bnb_bands: int | None,
        bsr_alpha: float | None,
        bsr_beta: float | None,
    ) -> list[float]:
        if explicit_coeffs:
            return list(explicit_coeffs)

        auto_bands = bsr_bands if noise_mechanism == 'bsr' else bnb_bands
        if auto_bands is None:
            raise ValueError(
                f'{noise_mechanism.upper()} auto coefficient generation requires '
                '--bsr-coeffs or bands (--bsr-bands for bsr, --bnb-bands for bnb).'
            )

        return generate_bsr_coeffs_from_sgd_workload(
            bands=int(auto_bands),
            momentum=float(bsr_beta if bsr_beta is not None else 0.0),
            weight_decay=float(bsr_alpha if bsr_alpha is not None else 1.0),
        )

    @staticmethod
    def _validate_correlated_mechanism_state(
        *,
        coeffs: list[float],
        z_std: float | None,
        sensitivity_scale: float | None,
    ) -> None:
        if not coeffs:
            raise ValueError('Correlated noise mechanism requires non-empty coeffs.')

        coeffs_f = [float(c) for c in coeffs]
        if not all(math.isfinite(c) for c in coeffs_f):
            raise ValueError('Correlated noise coeffs must be finite.')

        if coeffs_f[0] <= 1e-12:
            raise ValueError('Correlated noise requires coeffs[0] > 1e-12.')

        if z_std is not None and (not math.isfinite(float(z_std)) or float(z_std) < 0.0):
            raise ValueError('Correlated noise z_std must be finite and >= 0.')

        if sensitivity_scale is not None:
            s = float(sensitivity_scale)
            if (not math.isfinite(s)) or s <= 0.0:
                raise ValueError('Correlated noise sensitivity_scale must be finite and > 0.')

    @staticmethod
    def _resolve_expected_batch_size_for_correlated_runtime(
        *,
        total_steps: int | None,
        poisson_sampling: bool,
        sampling_mode: str | None,
        batch_size: int,
        dataset_size: int,
        dataloader_len: int,
        bnb_p: float | None,
        bnb_b: int | None,
    ) -> int:
        if total_steps:
            if not poisson_sampling and sampling_mode == 'b_min_sep':
                if bnb_p is None:
                    raise ValueError('b_min_sep sampling requires bnb_p to resolve expected batch size.')

                sample_rate = float(bnb_p)
            elif not poisson_sampling and sampling_mode == 'balls_in_bins':
                if bnb_b is None:
                    raise ValueError('balls_in_bins sampling requires bnb_b to resolve expected batch size.')

                sample_rate = 1.0 / float(int(bnb_b))
            else:
                sample_rate = float(batch_size) / float(dataset_size)
        else:
            sample_rate = 1.0 / float(dataloader_len)

        expected_batch_size = int(float(dataset_size) * float(sample_rate))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            expected_batch_size = int(
                expected_batch_size / int(torch.distributed.get_world_size())
            )

        return int(expected_batch_size)

    def setup(self):
        def _build_sampling_semantics() -> SamplingSemantics | None:
            if self.sampling_mode is None:
                return None

            privacy_metadata = {}
            if self.sampling_mode == 'cyclic_poisson' and self.bsr_bands is not None:
                privacy_metadata['bands'] = int(self.bsr_bands)

            if self.sampling_mode == 'b_min_sep':
                if self.bnb_b is not None:
                    privacy_metadata['b'] = int(self.bnb_b)

                if self.bnb_p is not None:
                    privacy_metadata['p'] = float(self.bnb_p)

                if self.bnb_bands is not None:
                    privacy_metadata['bands'] = int(self.bnb_bands)

            if self.sampling_mode == 'balls_in_bins':
                if self.bnb_b is not None:
                    privacy_metadata['bins'] = int(self.bnb_b)

                if self.bnb_bands is not None:
                    privacy_metadata['bands'] = int(self.bnb_bands)

            if self.bsr_max_participations is not None:
                privacy_metadata['max_participations'] = int(self.bsr_max_participations)

            if self.bsr_min_separation is not None:
                privacy_metadata['min_separation'] = int(self.bsr_min_separation)

            if self.bsr_mf_sensitivity is not None:
                privacy_metadata['mf_sensitivity'] = float(self.bsr_mf_sensitivity)

            if self.bsr_iterations_number is not None:
                privacy_metadata['iterations_number'] = int(self.bsr_iterations_number)

            return SamplingSemantics(
                sampling_mode=self.sampling_mode,
                privacy_metadata=privacy_metadata,
            )

        def _build_noise_mechanism_config(
            *,
            noise_multiplier_ref: float | None,
            bnb_horizon: int | None,
            bsr_cyclic_sensitivity_scale: float | None,
            correlated_denominator: float | None,
        ) -> NoiseMechanismConfig | None:
            if self.noise_mechanism not in ('bsr', 'bnb'):
                return None

            mechanism_state = {
                'coeffs': list(self.bsr_coeffs or []),
            }
            if (
                self.noise_mechanism == 'bsr'
                and self.sampling_mode == 'cyclic_poisson'
                and bsr_cyclic_sensitivity_scale is not None
            ):
                mechanism_state['sensitivity_scale'] = float(bsr_cyclic_sensitivity_scale)
            if self.noise_mechanism == 'bnb' and self.bnb_bands is not None:
                mechanism_state['bands'] = int(self.bnb_bands)
                if bnb_horizon is not None:
                    c_matrix, c_matrix_contract = build_bnb_toeplitz_c_matrix_and_contract(
                        coeffs=mechanism_state['coeffs'],
                        bands=int(self.bnb_bands),
                        horizon=int(bnb_horizon),
                    )
                    mechanism_state['c_matrix'] = c_matrix
                    mechanism_state['c_matrix_contract'] = c_matrix_contract

            if self.bsr_max_participations is not None:
                mechanism_state['max_participations'] = int(self.bsr_max_participations)

            if self.bsr_min_separation is not None:
                mechanism_state['min_separation'] = int(self.bsr_min_separation)

            if self.bsr_mf_sensitivity is not None:
                mechanism_state['mf_sensitivity'] = float(self.bsr_mf_sensitivity)
            if self.bsr_iterations_number is not None:
                mechanism_state['iterations_number'] = int(self.bsr_iterations_number)

            # make_private_with_epsilon calibrates and sets z_std itself.
            if not self._has_target_privacy_params():
                if self.bsr_z_std is not None:
                    mechanism_state['z_std'] = float(self.bsr_z_std)
                else:
                    if noise_multiplier_ref is None:
                        raise ValueError(
                            'BSR setup requires noise_multiplier when bsr_z_std is not provided.'
                        )
                    mechanism_state['z_std'] = calibrate_bsr_z_std(
                        noise_multiplier_ref=float(noise_multiplier_ref),
                        max_grad_norm=float(self.max_grad_norm),
                        denominator=float(correlated_denominator),
                    )

            self._validate_correlated_mechanism_state(
                coeffs=list(mechanism_state['coeffs']),
                z_std=mechanism_state.get('z_std'),
                sensitivity_scale=mechanism_state.get('sensitivity_scale'),
            )

            if self.noise_mechanism == 'bsr':
                return NoiseMechanismConfig(
                    mechanism='bsr',
                    accounting_mode=resolve_accounting_mode_from_accountant(self.accountant),
                    mechanism_state=mechanism_state,
                )

            return NoiseMechanismConfig(
                mechanism='bnb',
                accounting_mode=resolve_accounting_mode_from_accountant(self.accountant),
                mechanism_state=mechanism_state,
            )

        has_target_privacy_params = self._has_target_privacy_params()
        sampling_semantics = _build_sampling_semantics()

        if self.noise_mechanism in ('bsr', 'bnb'):
            had_explicit_coeffs = bool(self.bsr_coeffs)
            self.bsr_coeffs = self._resolve_or_generate_bsr_coeffs(
                noise_mechanism=self.noise_mechanism,
                explicit_coeffs=self.bsr_coeffs,
                bsr_bands=self.bsr_bands,
                bnb_bands=self.bnb_bands,
                bsr_alpha=self.bsr_alpha,
                bsr_beta=self.bsr_beta,
            )
            if not had_explicit_coeffs and torch.distributed.get_rank() == 0:
                auto_bands = self.bsr_bands if self.noise_mechanism == 'bsr' else self.bnb_bands
                log.info(
                    f'Auto-generated {self.noise_mechanism.upper()} coeffs from SGD workload '
                    f'(bands={auto_bands}, '
                    f'alpha={(self.bsr_alpha if self.bsr_alpha is not None else 1.0)}, '
                    f'beta={(self.bsr_beta if self.bsr_beta is not None else 0.0)}): '
                    f'{self.bsr_coeffs}'
                )

        noise_multiplier_ref = self.noise_multiplier
        if not has_target_privacy_params:
            if self.target_epsilon == -1:
                noise_multiplier_ref = 0.0

            elif self.noise_batch_ratio:
                noise_multiplier_ref = self.noise_batch_ratio * self.datamodule.batch_size

        train_dataloader = self.datamodule.get_dataloader('train')
        bsr_cyclic_horizon = None
        if self.noise_mechanism == 'bsr' and self.sampling_mode == 'cyclic_poisson':
            if self.total_steps:
                bsr_cyclic_horizon = int(self.total_steps)
            elif self.epochs:
                bsr_cyclic_horizon = int(self.epochs) * int(len(train_dataloader))

        bsr_cyclic_sensitivity_scale = self._resolve_bsr_cyclic_sensitivity_scale(
            coeffs=self.bsr_coeffs,
            steps=bsr_cyclic_horizon,
            iterations_number=self.bsr_iterations_number,
        )
        if (
            sampling_semantics is not None
            and self.noise_mechanism == 'bsr'
            and self.sampling_mode == 'cyclic_poisson'
            and bsr_cyclic_sensitivity_scale is not None
        ):
            sampling_semantics.privacy_metadata['sensitivity_scale'] = float(
                bsr_cyclic_sensitivity_scale
            )

        bnb_horizon = None
        if self.noise_mechanism == 'bnb':
            if self.total_steps:
                bnb_horizon = int(self.total_steps)
            elif self.epochs:
                bnb_horizon = int(self.epochs) * int(len(train_dataloader))

            if bnb_horizon is not None and bnb_horizon < 1:
                raise ValueError('BNB Toeplitz horizon must be >= 1.')

            if bnb_horizon is not None and self.bnb_bands is not None:
                bands = int(self.bnb_bands)
                if bands < 1:
                    raise ValueError('BNB bands must be >= 1.')
                if bnb_horizon % bands != 0:
                    aligned_horizon = int(math.ceil(bnb_horizon / bands) * bands)
                    if torch.distributed.get_rank() == 0:
                        log.info(
                            'Aligning BNB Toeplitz horizon to a multiple of bands '
                            f'for Monte Carlo accounting: horizon={bnb_horizon}, '
                            f'bands={bands}, aligned_horizon={aligned_horizon}.'
                        )
                    bnb_horizon = aligned_horizon

        correlated_denominator = None
        if self.noise_mechanism in ('bsr', 'bnb') and not has_target_privacy_params:
            expected_batch_size = self._resolve_expected_batch_size_for_correlated_runtime(
                total_steps=self.total_steps,
                poisson_sampling=self.poisson_sampling,
                sampling_mode=self.sampling_mode,
                batch_size=int(self.datamodule.batch_size),
                dataset_size=len(train_dataloader.dataset),
                dataloader_len=len(train_dataloader),
                bnb_p=self.bnb_p,
                bnb_b=self.bnb_b,
            )
            correlated_denominator = float(expected_batch_size)

        noise_mechanism_config = _build_noise_mechanism_config(
            noise_multiplier_ref=noise_multiplier_ref,
            bnb_horizon=bnb_horizon,
            bsr_cyclic_sensitivity_scale=bsr_cyclic_sensitivity_scale,
            correlated_denominator=correlated_denominator,
        )

        mechanism_kwargs = {}
        if self.bsr_mf_sensitivity is not None:
            mechanism_kwargs['bsr_mf_sensitivity'] = float(self.bsr_mf_sensitivity)

        if self.bsr_iterations_number is not None:
            mechanism_kwargs['bsr_iterations_number'] = int(self.bsr_iterations_number)

        if bsr_cyclic_sensitivity_scale is not None:
            mechanism_kwargs['bsr_sensitivity_scale'] = float(
                bsr_cyclic_sensitivity_scale
            )

        if self.bsr_max_participations is not None:
            mechanism_kwargs['bsr_max_participations'] = int(self.bsr_max_participations)

        if self.bsr_min_separation is not None:
            mechanism_kwargs['bsr_min_separation'] = int(self.bsr_min_separation)

        if self.bnb_bands is not None:
            mechanism_kwargs['bnb_bands'] = int(self.bnb_bands)

        if self.noise_mechanism == 'bnb' and has_target_privacy_params:
            mechanism_kwargs.update(
                resolve_bnb_calibration_kwargs(
                    overrides={
                        'bnb_num_samples': (
                            int(self.bnb_num_samples)
                            if self.bnb_num_samples is not None
                            else None
                        ),
                        'bnb_seed': (
                            int(self.bnb_seed)
                            if self.bnb_seed is not None
                            else None
                        ),
                    },
                )
            )

        self._log_bsr_trace(
            stage='dpdl_trainer_setup',
            sampling_semantics=sampling_semantics,
            noise_mechanism_config=noise_mechanism_config,
            has_target_privacy_params=has_target_privacy_params,
            noise_multiplier_ref=(
                float(noise_multiplier_ref)
                if noise_multiplier_ref is not None
                else None
            ),
            correlated_denominator=(
                float(correlated_denominator)
                if correlated_denominator is not None
                else None
            ),
            mechanism_kwargs=mechanism_kwargs,
        )

        noise_generator = torch.Generator(device=self.device)
        if self.seed:
            noise_generator.manual_seed(self.seed)

        self.model = self.model.to(self.device)

        # let's be distributed by default and wrap the model for Opacus DDP.
        # DifferentiallyPrivateDistributedDataParallel is actually a no-op in Opacus, but
        # let's wrap anyway in case of future api changes. https://opacus.ai/tutorials/ddp_tutorial
        model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(self.model)

        optimizer = self.optimizer

        # setup differential privacy for the model, optimize, and dataloader
        if has_target_privacy_params:
            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=self.epochs,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
                total_steps=self.total_steps,
                noise_mechanism_config=noise_mechanism_config,
                sampling_semantics=sampling_semantics,
                **mechanism_kwargs,
            )
        else:
            if self.target_epsilon == -1:
                self.noise_multiplier = 0

            if self.noise_batch_ratio:
                self.noise_multiplier = self.noise_batch_ratio * self.datamodule.batch_size

            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
                total_steps=self.total_steps,
                noise_mechanism_config=noise_mechanism_config,
                sampling_semantics=sampling_semantics,
            )

        # now we can start using the DP'ifyed stuff
        self.model = dp_model
        self.datamodule.set_dataloader('train', dp_dataloader)
        self.optimizer = dp_optimizer

    def get_epsilon(self):
        return self.privacy_engine.get_epsilon(self.target_delta)

    def _unwrap_model(self):
        # the model is wrapped inside Opacus, and Opacus distributed.
        # let's unwrap the vanilla model and return it
        return self.model._module.module

    def _fit_total_steps(self):
        # here we'll keep track of our approximate epochs
        virtual_epoch = 0

        # number of total steps taken
        step = 0

        # number of logical batches in an approximate epoch
        n_logical_batches = 0

        # track the logical batch loss here
        logical_batch_loss = 0

        # track the number of physical batches in a logical batch
        n_physical_batch_in_logical = 0

        # flag to indicate the beginning of a new logical batch
        logical_batch_begin = True

        # flag to indicate that a logical batch has been completed (set via the optimizer check)
        logical_batch_completed = False

        # to calculate the start/end of an epoch, we need the number
        # of steps in an epoch.
        steps_per_epoch = self._calculate_steps_per_epoch()
        progress_log_interval = max(1, min(100, steps_per_epoch // 5))
        stall_warning_interval = max(200, progress_log_interval * 10)
        max_physical_without_logical_step = max(5000, steps_per_epoch * 50)
        physical_since_last_logical_step = 0
        loop_start_time = time.monotonic()

        # At the very start, call on_train_batch_start for the first logical batch.
        if logical_batch_begin:
            self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
            logical_batch_begin = False

        self._handle_virtual_epoch_start(virtual_epoch)

        # Opacus-wrapped dataloaders can exhaust before `total_steps` under fixed-batch
        # semantics; keep reopening until we hit the target logical step count.
        while step < self.total_steps:
            batches_seen_in_pass = 0

            with BatchMemoryManager(
                data_loader=self.datamodule.get_dataloader('train'),
                max_physical_batch_size=self.physical_batch_size,
                optimizer=self.optimizer,
            ) as virtual_dataloader:
                for batch_idx, batch in enumerate(virtual_dataloader):
                    batches_seen_in_pass += 1
                    physical_since_last_logical_step += 1

                    # now, let's check if we are going to reach the end of logical batch.
                    # the optimizer will not skip next gradient update if we are not at
                    # the end of the logical batch. there's currently pretty much no other
                    # way to do it than this, because we don't know the size of the logical
                    # batch that was sampled.
                    if not self.optimizer._check_skip_next_step(False):
                        step += 1
                        logical_batch_completed = True
                    else:
                        logical_batch_completed = False

                    # notify the callbacks of a physical batch start
                    self.callback_handler.call('on_train_physical_batch_start', self, batch_idx, batch)

                    # let's fit this physical batch
                    batch_loss = self.fit_one_batch(batch_idx, batch)

                    # notify the callbacks of a physical batch end
                    self.callback_handler.call('on_train_physical_batch_end', self, batch_idx, batch, batch_loss)

                    # accumulate loss and count the number of physical batches in a logical batch
                    logical_batch_loss += batch_loss
                    n_physical_batch_in_logical += 1

                    # if the logical batch is complete, notify batch end and reset counters
                    if logical_batch_completed:
                        physical_since_last_logical_step = 0
                        self.callback_handler.call(
                            'on_train_batch_end',
                            self,
                            n_logical_batches,
                            None,
                            logical_batch_loss / n_physical_batch_in_logical,  # mean of physical batch losses
                        )
                        n_logical_batches += 1
                        logical_batch_loss = 0
                        n_physical_batch_in_logical = 0

                        # the next iteration starts a new logical batch
                        logical_batch_begin = True

                        if step == 1 or step == self.total_steps or step % progress_log_interval == 0:
                            elapsed = time.monotonic() - loop_start_time
                            progress_pct = 100.0 * float(step) / float(self.total_steps)
                            log.info(
                                'Training progress: logical step %s/%s (%.2f%%, virtual_epoch=%s, elapsed=%.1fs).',
                                step,
                                self.total_steps,
                                progress_pct,
                                virtual_epoch + 1,
                                elapsed,
                            )
                    else:
                        if physical_since_last_logical_step % stall_warning_interval == 0:
                            log.warning(
                                'No logical step completion for %s physical batches '
                                '(current step=%s/%s).',
                                physical_since_last_logical_step,
                                step,
                                self.total_steps,
                            )
                        if physical_since_last_logical_step >= max_physical_without_logical_step:
                            raise RuntimeError(
                                'Training made no logical-step progress for '
                                f'{physical_since_last_logical_step} physical batches. '
                                'Likely sampler/BatchMemoryManager mismatch or oversized '
                                'logical batches.'
                            )

                    # At the beginning of a new logical batch, call on_train_batch_start.
                    if logical_batch_begin:
                        self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
                        logical_batch_begin = False

                    # and next we check for epoch end
                    if (logical_batch_completed and step % steps_per_epoch == 0) or step == self.total_steps:
                        self._handle_virtual_epoch_end(virtual_epoch)

                        if self.validation_frequency and virtual_epoch % self.validation_frequency == 0:
                            # validate only on rank 0. no need to do distributed here,
                            # the computation is not heavy because we don't need gradients.
                            if torch.distributed.get_rank() == 0:
                                self.validate(virtual_epoch)

                            # other ranks will wait for validation
                            torch.distributed.barrier()

                        if step < self.total_steps:
                            virtual_epoch += 1
                            self._handle_virtual_epoch_start(virtual_epoch)
                            # Start a new logical batch for the new epoch.
                            self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
                            logical_batch_begin = False
                        else:
                            break

                    # Reset the logical batch completion flag for the next iteration.
                    logical_batch_completed = False

            if batches_seen_in_pass == 0:
                raise RuntimeError(
                    'Train dataloader yielded zero batches while using total_steps mode.'
                )

        if step != self.total_steps:
            log.warning(f'Was going to step for {self.total_steps}, but stepped only {step} steps.')

    def fit_one_batch(self, batch_idx, batch):
        self.optimizer.zero_grad()

        X, y = batch
        X, y = self.adapter.move_to_device(X, y)

        forward_output = self.adapter.forward(self._unwrap_model(), (X, y))
        loss = self.adapter.compute_loss(self._unwrap_model(), (X, y), forward_output, normalize_by=None)
        self.adapter.update_metrics(self._unwrap_model(), (X, y), forward_output)
        loss.backward()

        self.optimizer.step()

        loss = loss.item()

        return loss

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        logical_idx = 0
        logical_loss = 0.0
        phys_in_logical = 0
        in_new_logical = True

        with BatchMemoryManager(
            data_loader=self.datamodule.get_dataloader('train'),
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer,
        ) as virtual_dataloader:

            for phys_idx, batch in enumerate(virtual_dataloader):

                # if we're starting a new logical batch, signal start
                if in_new_logical:
                    self.callback_handler.call(
                        'on_train_batch_start', self, logical_idx, None
                    )
                    in_new_logical = False

                # physical‐batch callbacks
                self.callback_handler.call(
                    'on_train_physical_batch_start', self, phys_idx, batch
                )

                loss = self.fit_one_batch(phys_idx, batch)

                self.callback_handler.call(
                    'on_train_physical_batch_end', self, phys_idx, batch, loss
                )

                # accumulate
                logical_loss += loss
                phys_in_logical += 1

                # check for logical‐batch boundary
                if not self.optimizer._check_skip_next_step(False):
                    avg = logical_loss / phys_in_logical
                    self.callback_handler.call(
                        'on_train_batch_end',
                        self,
                        logical_idx,
                        None,
                        avg,
                    )
                    logical_idx += 1
                    logical_loss = 0.0
                    phys_in_logical = 0
                    in_new_logical = True

        # wrap up epoch
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()
        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)


class TaskAdapter:
    """
        Adapter class for different Tasks.

        One adapter per task family: classification, Causal-LM, ..

        These are to follow the open/close principle: instead of changing
        the Trainer(s), we can just create a new adapter for a new task.

        Handles per-task splitting, moving to device, forward/loss/metrics calls.
    """
    def __init__(self, device: torch.device):
        self.device = device

    def move_to_device(self, X, y=None):
        device = self.device

        def move(obj):
            if isinstance(obj, Mapping):
                return {k: move(v) for k, v in obj.items()}
            elif isinstance(obj, torch.Tensor):
                return obj.to(device=device, non_blocking=True)
            else:
                return obj

        X = move(X)
        y = move(y) if y is not None else None

        return (X, y) if y is not None else X

    def iterate_physical_batches(self, batch, physical_batch_size):
        """
        Return an iterator over physical batches.
        """
        ...

    def forward(self, model, batch):
        raise NotImplementedError

    def compute_loss(self, model, batch, forward_output, normalize_by: int | None = None):
        raise NotImplementedError

    def update_metrics(self, model, batch, forward_output, metrics = None):
        raise NotImplementedError


class ClassificationAdapter(TaskAdapter):
    def iterate_physical_batches(self, batch, physical_batch_size):
        X, y = batch
        for Xs, ys in zip(X.split(physical_batch_size, 0), y.split(physical_batch_size, 0)):
            yield (Xs, ys)

    def forward(self, model, batch):
        X, _ = batch
        logits = model(X)
        return logits

    def compute_loss(self, model, batch, forward_output, normalize_by: int | None = None):
        _, y = batch

        loss = model.criterion(forward_output, y)

        if normalize_by:
            loss = loss / normalize_by

        return loss

    def update_metrics(self, model, batch, forward_output, metrics = None):
        _, y = batch

        if metrics is not None:
            metrics_to_update = metrics
        else:
            metrics_to_update = model.train_metrics if model.training else model.valid_metrics

        preds = torch.argmax(forward_output, dim=1)
        metrics_to_update.update(preds, y)


class LanguageModelAdapter(TaskAdapter):
    def iterate_physical_batches(self, batch, physical_batch_size):
        X, y = batch
        splits = {k: v.split(physical_batch_size, dim=0) for k, v in X.items()}
        y_splits = y.split(physical_batch_size, dim=0)

        for i in range(len(y_splits)):
            yield ({k: splits[k][i] for k in splits}, y_splits[i])

    def forward(self, model, batch):
        X, _ = batch
        logits = model(X)
        return logits

    def compute_loss(self, model, batch, forward_output, normalize_by: int | None = None):
        _, y = batch
        preds, y_flat = shift_and_flatten(forward_output, y)

        loss = model.criterion(preds, y_flat)

        if normalize_by:
            loss = loss / normalize_by

        return loss

    def update_metrics(self, model, batch, forward_output, metrics = None):
        _, y = batch
        if metrics is not None:
            metrics_to_update = metrics
        else:
            metrics_to_update = model.train_metrics if model.training else model.valid_metrics

        with torch.no_grad():
            metrics_to_update.update(forward_output, y)

# Define task specific adapters
_ADAPTERS = {
    'ImageClassification': ClassificationAdapter,
    'SequenceClassification': ClassificationAdapter,
    'CausalLM': LanguageModelAdapter,
    'InstructLM': LanguageModelAdapter,
}

class TrainerFactory:

    @staticmethod
    def _make_adapter(configuration, device):
        task = configuration.task or 'classification'

        if task not in _ADAPTERS:
            raise ValueError(f'No adapter for task "{task}"')

        return _ADAPTERS[task](device)

    @staticmethod
    def get_trainer(config_manager: ConfigurationManager) -> Trainer:
        device = resolve_device(config_manager.configuration.device)

        # are we differentially private?
        if config_manager.configuration.privacy:
            return TrainerFactory._get_differentially_private_trainer(
                config_manager.configuration,
                config_manager.hyperparams,
                device,
            )

        # XXX: checkpoint dir??? We should have this from the new save model implementation?
        if config_manager.configuration.checkpoint_step_interval is not None:
            config_manager.configuration.checkpoints_dir = os.path.join(
                config_manager.configuration.log_dir,
                config_manager.configuration.experiment_name,
                'checkpoints',
        )

        return TrainerFactory._get_basic_trainer(
            config_manager.configuration,
            config_manager.hyperparams,
            device,
        )

    @staticmethod
    def _get_basic_trainer(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        device: torch.device,
    ) -> Trainer:

        # First create DataModule, it can figure out the number of classes
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams, device)
        num_classes = datamodule.get_num_classes()

        # Now, setup data, model, and optimizer
        loss_fn = LossFactory.get_loss(configuration)

        # This also return effective number of classes, as for LM tasks
        # it is vocabulary size and for classification tasksk it's number
        # of classes as usually.
        model, transforms, num_classes_eff = ModelFactory.get_model(
            configuration,
            hyperparams,
            num_classes,
            loss_fn,
        )

        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)
        metrics = MetricsFactory.get_metrics(configuration, num_classes_eff)
        model.set_metrics(metrics)

        # Initialize the datamodule with the transformations
        datamodule.initialize(transforms)

        # should we cache outputs from the feature extractor?
        if configuration.cache_features:
            # compute cache on rank 0 only
            if torch.distributed.get_rank() == 0:
                datamodule.cache_features(model)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                datamodule.cache_features(model)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams, device=device)
        )

        epochs, total_steps = TrainerFactory._get_epochs_and_steps(configuration, hyperparams, datamodule)

        adapter = TrainerFactory._make_adapter(configuration, device)

        # instantiate a trainer without dp
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            adapter=adapter,
            callback_handler=callback_handler,
            physical_batch_size=configuration.physical_batch_size,
            epochs=epochs,
            total_steps=total_steps,
            seed=configuration.seed,
            validation_frequency=configuration.validation_frequency,
            peft=configuration.peft,
            task=configuration.task,
            device=device,
        )

        return trainer

    @staticmethod
    def _get_differentially_private_trainer(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        device: torch.device,
    ) -> Trainer:
        # Target delta calculation: A common heuristic is to use 1/N', with N'
        # being the size of the dataset rounded up to the nearest power of 10.
        # To avoid too large values of delta, let's pick a somewhat sensible
        # minimum of 1e-5.
        def _round_up_to_nearest_power_of_10(n):
            return 10 ** math.ceil(math.log10(n))

        def _calculate_target_delta(N):
            N_prime = _round_up_to_nearest_power_of_10(N)
            return min(1e-5, 1 / N_prime)

        def _get_target_privacy_params(hyperparams):
            N = len(datamodule.get_dataloader('train').dataset)
            target_delta = _calculate_target_delta(N)

            if torch.distributed.get_rank() == 0:
                log.info(f'Dataset size is {N}, setting target delta to: {target_delta}.')

            # are we given a target epsilon?
            if hyperparams.target_epsilon is not None:
                target_epsilon = hyperparams.target_epsilon
            else:
                target_epsilon = None

            return target_delta, target_epsilon

        # First initialize the DataModule, it will know about the number of classes
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams, device)
        num_classes = datamodule.get_num_classes()

        # Now, setup data, model, and optimizer
        loss_fn = LossFactory.get_loss(configuration)

        model, transforms, num_classes_eff = ModelFactory.get_model(
            configuration,
            hyperparams,
            num_classes,
            loss_fn,
        )

        metrics = MetricsFactory.get_metrics(configuration, num_classes_eff)
        model.set_metrics(metrics)

        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)

        # The datamodule needs to be aware of the transformations, now we can initialize it
        datamodule.initialize(transforms)
        dataloader = datamodule.get_dataloader('train')

        # Are we caching the outputs of the feature extractor
        if configuration.cache_features:
            # compute cache on rank 0 only
            if torch.distributed.get_rank() == 0:
                datamodule.cache_features(model)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                datamodule.cache_features(model)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams, device=device)
        )

        target_delta, target_epsilon = _get_target_privacy_params(hyperparams)
        epochs, total_steps = TrainerFactory._get_epochs_and_steps(configuration, hyperparams, datamodule)

        adapter = TrainerFactory._make_adapter(configuration, device)

        # instantiate a differentialy private trained
        trainer = DifferentiallyPrivateTrainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            adapter=adapter,
            # hypers
            epochs=epochs,
            total_steps=total_steps,
            noise_multiplier=hyperparams.noise_multiplier,
            max_grad_norm=hyperparams.max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            noise_batch_ratio=hyperparams.noise_batch_ratio,
            poisson_sampling=configuration.poisson_sampling,
            normalize_clipping=configuration.normalize_clipping,
            noise_mechanism=configuration.noise_mechanism,
            sampling_mode=configuration.sampling_mode,
            bsr_coeffs=configuration.bsr_coeffs,
            bsr_z_std=configuration.bsr_z_std,
            bsr_bands=configuration.bsr_bands,
            bsr_max_participations=configuration.bsr_max_participations,
            bsr_min_separation=configuration.bsr_min_separation,
            bsr_mf_sensitivity=configuration.bsr_mf_sensitivity,
            bsr_iterations_number=configuration.bsr_iterations_number,
            bsr_alpha=configuration.bsr_alpha,
            bsr_beta=configuration.bsr_beta,
            bnb_b=configuration.bnb_b,
            bnb_p=configuration.bnb_p,
            bnb_bands=configuration.bnb_bands,
            bnb_num_samples=configuration.bnb_num_samples,
            bnb_seed=configuration.bnb_seed,
            # config
            accountant=configuration.accountant,
            secure_mode=configuration.secure_mode,
            clipping_mode=configuration.clipping_mode,
            physical_batch_size=configuration.physical_batch_size,
            seed=configuration.seed,
            callback_handler=callback_handler,
            validation_frequency=configuration.validation_frequency,
            peft=configuration.peft,
            task=configuration.task,
            device=device,
        )

        return trainer

    @staticmethod
    def _get_epochs_and_steps(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        datamodule: DataModule,
    ):
        """
        Compute the number of training epochs and total optimizer steps.

        If `use_steps=True`, we convert epochs to total_steps using ceil(N / B),
        which matches the default logic in Opacus:
            - sample_rate = 1 / ceil(N / B)
            - steps = int(1 / sample_rate) = ceil(N / B)

        However, default Opacus might still make more steps than us, because we
        cap the total number of steps exactly at `total_steps` and Opacus default
        (`use_steps=False`) always makes a full pass on the dataloader when feeding
        batches through the BatchMemoryManager.

        Returns:
            (epochs, total_steps): One of the values will be None depending on mode.
        """

        # If we're using step-based training and the number of epochs is specified,
        # convert epochs to total steps using the default Opacus logic.
        if configuration.use_steps and hyperparams.epochs:
            dataloader = datamodule.get_dataloader('train')

            # Match Opacus: steps_per_epoch = ceil(N / B)
            N = len(dataloader.dataset)
            B = datamodule.batch_size
            steps_per_epoch = math.ceil(N / B)
            total_steps = steps_per_epoch * hyperparams.epochs
            epochs = None

        # If total steps are manually specified in config
        elif configuration.use_steps and hyperparams.total_steps:
            total_steps = hyperparams.total_steps
            epochs = None

        # Standard epoch-based training
        else:
            total_steps = None
            epochs = hyperparams.epochs

        return epochs, total_steps
