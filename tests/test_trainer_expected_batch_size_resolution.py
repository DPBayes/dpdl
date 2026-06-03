import pytest
import torch

pytest.importorskip('torch')
pytest.importorskip('opacus')

from dpdl.trainer import DifferentiallyPrivateTrainer
from integration_utils import load_json


def test_fourier_telemetry_summary_reports_compression_and_rank_consistency() -> None:
    per_rank = [
        {
            'rank': 0,
            'metadata': {
                'original_trainable_numel': 100,
                'encoded_numel': 25,
                'encoded_fraction_of_original': 0.25,
                'compression_ratio_vs_original': 4.0,
                'plan_entry_count': 3,
                'fallback_counts': {'parameter_blockwise': 1},
                'plan_kind_counts': {'matrix_left': 2, 'blockwise': 1},
            },
        },
        {
            'rank': 1,
            'metadata': {
                'original_trainable_numel': 100,
                'encoded_numel': 25,
                'encoded_fraction_of_original': 0.25,
                'compression_ratio_vs_original': 4.0,
                'plan_entry_count': 3,
                'fallback_counts': {'parameter_blockwise': 1},
                'plan_kind_counts': {'matrix_left': 2, 'blockwise': 1},
            },
        },
    ]

    summary = DifferentiallyPrivateTrainer._summarize_fourier_rank_payloads(per_rank)

    assert summary['consistent_dimensions_across_ranks'] is True
    assert summary['original_trainable_numel'] == 100
    assert summary['encoded_numel'] == 25
    assert summary['encoded_fraction_of_original'] == 0.25
    assert summary['compression_ratio_vs_original'] == 4.0
    assert summary['fallback_counts'] == {'parameter_blockwise': 1}


def test_fourier_telemetry_summary_flags_rank_dimension_mismatch() -> None:
    per_rank = [
        {'rank': 0, 'metadata': {'original_trainable_numel': 100, 'encoded_numel': 25}},
        {'rank': 1, 'metadata': {'original_trainable_numel': 100, 'encoded_numel': 50}},
    ]

    summary = DifferentiallyPrivateTrainer._summarize_fourier_rank_payloads(per_rank)

    assert summary['consistent_dimensions_across_ranks'] is False


def test_memory_telemetry_summary_uses_global_peak_across_ranks() -> None:
    per_rank = [
        {
            'rank': 0,
            'cuda_memory_available': True,
            'peak_memory_allocated_bytes': 10,
            'peak_memory_reserved_bytes': 20,
            'current_memory_allocated_bytes': 3,
            'current_memory_reserved_bytes': 4,
        },
        {
            'rank': 1,
            'cuda_memory_available': True,
            'peak_memory_allocated_bytes': 30,
            'peak_memory_reserved_bytes': 15,
            'current_memory_allocated_bytes': 7,
            'current_memory_reserved_bytes': 2,
        },
    ]

    summary = DifferentiallyPrivateTrainer._summarize_memory_rank_payloads(per_rank)

    assert summary['cuda_rank_count'] == 2
    assert summary['max_peak_memory_allocated_bytes'] == 30
    assert summary['max_peak_memory_reserved_bytes'] == 20
    assert summary['max_current_memory_allocated_bytes'] == 7
    assert summary['max_current_memory_reserved_bytes'] == 4


def test_fourier_and_memory_telemetry_are_written_to_experiment_dir(tmp_path) -> None:
    class FakeOptimizer:
        fourier_clipping_metadata = {
            'original_trainable_numel': 100,
            'encoded_numel': 20,
            'encoded_fraction_of_original': 0.2,
            'compression_ratio_vs_original': 5.0,
            'plan_entry_count': 1,
            'fallback_counts': {},
            'plan_kind_counts': {'matrix_left': 1},
        }

    trainer = object.__new__(DifferentiallyPrivateTrainer)
    trainer.log_dir = str(tmp_path)
    trainer.experiment_name = 'telemetry'
    trainer.clipping_mode = 'fourier'
    trainer.noise_mechanism = 'gaussian'
    trainer.optimizer = FakeOptimizer()
    trainer.device = torch.device('cpu')
    trainer._memory_telemetry_reset_error = None

    trainer._emit_fourier_telemetry(phase='post_fit')
    trainer._emit_memory_telemetry(phase='post_fit')

    fourier_payload = load_json(tmp_path / 'telemetry' / 'fourier_telemetry.json')
    memory_payload = load_json(tmp_path / 'telemetry' / 'memory_telemetry.json')

    assert fourier_payload['summary']['original_trainable_numel'] == 100
    assert fourier_payload['summary']['encoded_numel'] == 20
    assert fourier_payload['summary']['compression_ratio_vs_original'] == 5.0
    assert memory_payload['summary']['cuda_rank_count'] == 0
    assert memory_payload['summary']['max_peak_memory_allocated_bytes'] is None


def test_expected_batch_size_resolution_total_steps_b_min_sep() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=100,
        poisson_sampling=False,
        sampling_mode='b_min_sep',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=0.25,
        bnb_b=2,
    )
    assert got == 12


def test_expected_batch_size_resolution_total_steps_balls_in_bins() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=100,
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=8,
    )
    assert got == 8


def test_expected_batch_size_resolution_total_steps_balls_in_bins_defaults_bins_from_steps_per_epoch() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=100,
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
    )
    assert got == 4


def test_expected_batch_size_resolution_total_steps_torch_sampler_bsr() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=100,
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
    )
    assert got == 4


def test_expected_batch_size_resolution_epochs_path() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=None,
        poisson_sampling=False,
        sampling_mode='torch_sampler',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
    )
    assert got == 4


def test_expected_batch_size_resolution_requires_sampler_inputs() -> None:
    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=10,
        poisson_sampling=False,
        sampling_mode='b_min_sep',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=2,
    )
    assert got == 4

    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=10,
        poisson_sampling=False,
        sampling_mode='b_min_sep',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
        bsr_bands=4,
    )
    assert got == 4

    with pytest.raises(ValueError, match='requires bnb_b or a derivable band parameter'):
        DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
            total_steps=10,
            poisson_sampling=False,
            sampling_mode='b_min_sep',
            batch_size=4,
            dataset_size=64,
            dataloader_len=16,
            bnb_p=None,
            bnb_b=None,
        )

    got = DifferentiallyPrivateTrainer._resolve_expected_batch_size_for_correlated_runtime(
        total_steps=10,
        poisson_sampling=False,
        sampling_mode='balls_in_bins',
        batch_size=4,
        dataset_size=64,
        dataloader_len=16,
        bnb_p=None,
        bnb_b=None,
    )
    assert got == 4
