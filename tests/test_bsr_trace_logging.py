from __future__ import annotations

import json
import logging

from opacus.mechanism_contracts import NoiseMechanismConfig, SamplingSemantics

from dpdl.configurationmanager import ConfigurationManager
from dpdl.trainer import DifferentiallyPrivateTrainer


def test_configuration_manager_emits_bsr_trace(caplog) -> None:
    cli_params = {
        'command': 'train',
        'privacy': True,
        'epochs': 1,
        'batch_size': 32,
        'target_epsilon': 8.0,
        'noise_multiplier': None,
        'noise_batch_ratio': None,
        'max_grad_norm': 1.0,
        'noise_mechanism': 'bsr',
        'accountant': 'bsr',
        'poisson_sampling': False,
        'sampling_mode': 'cyclic_poisson',
        'bsr_bands': 100,
    }

    with caplog.at_level(logging.INFO, logger='dpdl.configurationmanager'):
        ConfigurationManager(cli_params)

    records = [r.message for r in caplog.records if r.message.startswith('BSR_TRACE ')]
    assert len(records) == 1
    payload = json.loads(records[0][len('BSR_TRACE '):])
    assert payload['stage'] == 'dpdl_config_parse'
    assert payload['noise_mechanism'] == 'bsr'
    assert payload['bsr']['bands'] == 100


def test_configuration_manager_does_not_emit_bsr_trace_for_non_bsr(caplog) -> None:
    cli_params = {
        'command': 'train',
        'privacy': True,
        'epochs': 1,
        'batch_size': 32,
        'noise_multiplier': 1.0,
        'target_epsilon': None,
        'noise_batch_ratio': None,
        'max_grad_norm': 1.0,
        'noise_mechanism': 'gaussian',
        'accountant': 'prv',
    }

    with caplog.at_level(logging.INFO, logger='dpdl.configurationmanager'):
        ConfigurationManager(cli_params)

    assert not [r for r in caplog.records if r.message.startswith('BSR_TRACE ')]


def test_trainer_bsr_trace_helper_emits_payload(caplog) -> None:
    config = NoiseMechanismConfig(
        mechanism='bsr',
        accounting_mode='bsr_accountant',
        mechanism_state={
            'coeffs': [1.0, 0.5, 0.25],
            'z_std': 0.02,
            'sensitivity_scale': 1.1,
        },
    )
    semantics = SamplingSemantics(
        sampling_mode='cyclic_poisson',
        privacy_metadata={'bands': 100, 'sample_rate': 0.01},
    )

    with caplog.at_level(logging.INFO, logger='dpdl.trainer'):
        DifferentiallyPrivateTrainer._log_bsr_trace(
            stage='unit_test',
            sampling_semantics=semantics,
            noise_mechanism_config=config,
            has_target_privacy_params=True,
            noise_multiplier_ref=1.23,
            correlated_denominator=500.0,
            mechanism_kwargs={'bsr_sensitivity_scale': 1.1},
        )

    records = [r.message for r in caplog.records if r.message.startswith('BSR_TRACE ')]
    assert len(records) == 1
    payload = json.loads(records[0][len('BSR_TRACE '):])
    assert payload['stage'] == 'unit_test'
    assert payload['mechanism'] == 'bsr'
    assert payload['mechanism_state']['coeff_count'] == 3
