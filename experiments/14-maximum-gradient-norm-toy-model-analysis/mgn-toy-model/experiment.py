import time
import pickle
from functools import partial
import optuna
from optuna.integration import BoTorchSampler
from data_model import MeanEstimator, create_data_loaders
from train import train_dp_model
from plotting import plot_all_repeats, plot_mean_with_confidence_intervals
from torch.utils.data import DataLoader
from utils import seed_everything
import sys


def optimize_hyperparameters(
    train_dataset,
    val_dataset,
    max_grad_norm,
    n_trials=50,
    epochs=10,
    epsilon=0.25,
    input_dim=10,
    seed=42,
):
    """
    Optimize hyperparameters for a given max_grad_norm using Optuna with seed control.
    """
    # Set seed for reproducibility
    seed_everything(seed)

    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1, log=True)
        batch_size = trial.suggest_int('batch_size', 8, 1000)

        # Re-create data loaders with the suggested batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        # Initialize model
        model = MeanEstimator(input_dim)

        # Train and evaluate the model
        result = train_dp_model(
            model,
            train_loader,
            val_loader,
            epochs,
            learning_rate,
            max_grad_norm,
            epsilon,
            seed,
        )

        # Return the final validation loss to be minimized
        return result['val_losses'][-1]

    # Use the BoTorchSampler with a fixed seed for consistent sampling
    sampler = BoTorchSampler(seed=seed)

    # Create an Optuna study
    study = optuna.create_study(direction='minimize', sampler=sampler)

    # Optimize the study
    study.optimize(objective, n_trials=n_trials)

    # Get the best trial
    best_trial = study.best_trial

    return best_trial.value, best_trial.params


def run_repeats_with_optimized_hypers(
    max_grad_norms,
    optimized_params,
    train_dataset,
    val_dataset,
    repeats=200,
    epochs=10,
    epsilon=0.25,
    input_dim=10,
    seed=42,
):
    """
    Run repeated experiments using optimized hyperparameters for different max_grad_norms.
    """
    all_results_dict = {mgn: [] for mgn in max_grad_norms}
    all_results_list = []

    for repeat in range(repeats):
        print(f'Running repeat {repeat + 1} of {repeats}...', file=sys.stderr)
        seed_everything(repeat)

        repeat_losses = []
        for idx, max_grad_norm in enumerate(max_grad_norms):
            params = optimized_params[idx]
            learning_rate = params['learning_rate']
            batch_size = params['batch_size']

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=len(val_dataset), shuffle=False
            )

            model = MeanEstimator(input_dim)
            result = train_dp_model(
                model,
                train_loader,
                val_loader,
                epochs,
                learning_rate,
                max_grad_norm,
                epsilon,
                seed,
            )
            all_results_dict[max_grad_norm].append(result['val_losses'][-1])
            repeat_losses.append(result['val_losses'][-1])

        all_results_list.append(repeat_losses)

    return all_results_dict, all_results_list


def run_single_optimized_experiment(
    max_grad_norms,
    optimized_params,
    train_dataset,
    val_dataset,
    epochs=10,
    epsilon=0.25,
    input_dim=10,
    seed=42,
):
    """
    Run a single repeat with optimized hyperparameters, tracking losses, clipped proportions, and predicted means.
    """

    print(
        f'Collecting losses, clipped proportions, and predicted means using optimized hypers and epsilon {epsilon}...'
    )

    train_results_dict = {}
    val_results_dict = {}
    clipped_proportions_dict = {}
    train_means_dict = {}
    val_means_dict = {}

    results = []
    for idx, max_grad_norm in enumerate(max_grad_norms):
        print(f' - ITER MGN: {max_grad_norm}')
        params = optimized_params[idx]
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        model = MeanEstimator(input_dim)
        result = train_dp_model(
            model,
            train_loader,
            val_loader,
            epochs,
            learning_rate,
            max_grad_norm,
            epsilon,
            seed,
        )

        results.append(
            (
                result['train_losses'],
                result['val_losses'],
                result['clipped_proportions'],
                result['predicted_means_norm_train'],
                result['predicted_means_norm_val'],
            )
        )

    # Unpack results into separate lists for train losses, val losses, and clipped proportions
    (
        train_losses,
        val_losses,
        clipped_proportions,
        train_mean_norms,
        val_mean_norms,
    ) = zip(*results)

    train_results_dict[epsilon] = train_losses
    val_results_dict[epsilon] = val_losses
    clipped_proportions_dict[epsilon] = clipped_proportions
    train_means_dict[epsilon] = train_mean_norms
    val_means_dict[epsilon] = val_mean_norms

    return (
        train_results_dict,
        val_results_dict,
        clipped_proportions_dict,
        train_means_dict,
        val_means_dict,
    )


def save_results(filename, data):
    """
    Save experiment results to a file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_results(filename):
    """
    Load experiment results from a file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
