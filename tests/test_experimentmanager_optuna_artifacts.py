import json
from types import SimpleNamespace

from dpdl.experimentmanager import save_optuna_input_artifacts


def _config_manager(**overrides):
    defaults = {
        'command': 'optimize',
        'optuna_config': None,
        'optuna_manual_trials': None,
        'target_hypers': ['learning_rate'],
        'n_trials': 25,
        'optuna_sampler': 'RandomSampler',
        'optuna_direction': 'maximize',
        'optuna_target_metric': 'MulticlassAccuracy',
        'optuna_random_trials': 10,
        'optuna_resume': True,
        'optuna_journal': 'optuna.journal',
    }
    defaults.update(overrides)
    return SimpleNamespace(configuration=SimpleNamespace(**defaults))


def test_save_optuna_input_artifacts_snapshots_configs_without_overwrite(tmp_path):
    optuna_config = tmp_path / 'search-space.conf'
    manual_trials = tmp_path / 'manual-trials.conf'
    experiment_dir = tmp_path / 'experiment'
    experiment_dir.mkdir()

    optuna_config.write_text(
        'learning_rate:\n'
        '  min: 1e-5\n'
        '  max: 0.1\n'
        '  type: float\n'
        '  log_space: true\n'
    )
    manual_trials.write_text('trials:\n  - learning_rate: 0.001\n')

    save_optuna_input_artifacts(
        _config_manager(
            optuna_config=str(optuna_config),
            optuna_manual_trials=str(manual_trials),
        ),
        experiment_dir,
    )

    assert (experiment_dir / 'optuna.conf').read_text() == optuna_config.read_text()
    assert (experiment_dir / 'optuna-manual-trials.conf').read_text() == manual_trials.read_text()

    metadata = json.loads((experiment_dir / 'optuna-inputs.json').read_text())
    assert metadata['optuna_config'] == str(optuna_config)
    assert metadata['optuna_config_artifact'] == 'optuna.conf'
    assert metadata['optuna_manual_trials'] == str(manual_trials)
    assert metadata['optuna_manual_trials_artifact'] == 'optuna-manual-trials.conf'
    assert metadata['target_hypers'] == ['learning_rate']
    assert metadata['n_trials'] == 25
    assert metadata['optuna_resume'] is True

    original_snapshot = (experiment_dir / 'optuna.conf').read_text()
    optuna_config.write_text('learning_rate:\n  min: 1e-1\n  max: 1.0\n')

    save_optuna_input_artifacts(
        _config_manager(optuna_config=str(optuna_config)),
        experiment_dir,
    )

    assert (experiment_dir / 'optuna.conf').read_text() == original_snapshot


def test_save_optuna_input_artifacts_ignores_non_optimize_commands(tmp_path):
    optuna_config = tmp_path / 'search-space.conf'
    experiment_dir = tmp_path / 'experiment'
    experiment_dir.mkdir()
    optuna_config.write_text('learning_rate:\n  min: 1e-5\n')

    save_optuna_input_artifacts(
        _config_manager(command='train', optuna_config=str(optuna_config)),
        experiment_dir,
    )

    assert not (experiment_dir / 'optuna.conf').exists()
    assert not (experiment_dir / 'optuna-inputs.json').exists()
