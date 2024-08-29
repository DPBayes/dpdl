import torch
import numpy as np
import argparse
from experiment import (
    optimize_hyperparameters,
    run_repeats_with_optimized_hypers,
    run_single_optimized_experiment,
    save_results,
    load_results,
)
from data_model import create_data_loaders, TensorDataset, random_split
from utils import seed_everything, generate_mixture_data, setup_directories
from plotting import (
    plot_all_repeats,
    plot_mean_with_confidence_intervals,
    plot_clipped_proportions_per_iteration,
    plot_last_clipped_proportion_vs_mgn,
    plot_predicted_means_all_epsilons,
)

# Function to optimize hyperparameters for all max_grad_norms
def optimize_for_all_max_grad_norms(train_dataset, val_dataset, max_grad_norms, epsilon, input_dim, epochs, seed):
    optimized_losses = []
    optimized_params = []

    for max_grad_norm in max_grad_norms:
        print(f'Optimizing hyperparameters for max_grad_norm = {max_grad_norm}')
        best_loss, best_params = optimize_hyperparameters(
            train_dataset,
            val_dataset,
            max_grad_norm,
            n_trials=150,
            epochs=epochs,
            epsilon=epsilon,
            input_dim=input_dim,
            seed=seed,
        )
        optimized_losses.append(best_loss)
        optimized_params.append(best_params)

    return optimized_losses, optimized_params

# Function to handle plotting of results
def plot_experiment_results(max_grad_norms, all_results_list, all_results_dict, repeats, epsilon, image_dir):
    plot_all_repeats(
        max_grad_norms,
        all_results_list,
        repeats,
        epsilon,
        save_fname=f'{image_dir}/toy-model-repeats-{repeats}-with-optimized-hypers-epsilon-{epsilon}.png',
    )
    plot_mean_with_confidence_intervals(
        max_grad_norms,
        all_results_dict,
        repeats,
        epsilon,
        save_fname=f'{image_dir}/toy-model-repeats-{repeats}-with-optimized-hypers-epsilon-{epsilon}-mean-with-confidence.png',
    )

# Main function to handle the experiment logic
def main(data_dir='toy-model-data', image_dir='temp', seed=42, split_seed=42):
    # Constants for the experiment
    D = 10  # Dimensionality of the data
    N = 1250  # Number of data points
    EPOCHS = 10
    REPEATS = 50
    EPSILON = 0.25

    lower_bound = 0.3
    upper_bound = 4
    max_grad_norms = np.geomspace(lower_bound, upper_bound, 15)

    print(f'Running experiment with seed = {seed} and split_seed = {split_seed}')

    # Set up directories
    current_data_dir = f'{data_dir}-seed{seed}'
    current_image_dir = f'{image_dir}-seed{seed}'
    setup_directories(current_data_dir, current_image_dir)

    # Seed everything
    seed_everything(split_seed)

    # Generate data and split
    data = generate_mixture_data(N, D)
    print(f'DATA MEAN: {data.mean()}')

    dataset = TensorDataset(data)
    train_size = int(0.8 * N)
    val_size = N - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Optimize hyperparameters
    seed_everything(seed)
    optimized_losses, optimized_params = optimize_for_all_max_grad_norms(
        train_dataset, val_dataset, max_grad_norms, EPSILON, D, EPOCHS, seed
    )

    # Save optimized parameters
    save_results(f'{current_data_dir}/optimized_params.pkl', optimized_params)
    save_results(f'{current_data_dir}/optimized_losses.pkl', optimized_losses)

    # Run repeated experiments
    all_results_dict, all_results_list = run_repeats_with_optimized_hypers(
        max_grad_norms,
        optimized_params,
        train_dataset,
        val_dataset,
        repeats=REPEATS,
        epochs=EPOCHS,
        epsilon=EPSILON,
        input_dim=D,
        seed=seed,
    )

    # Save results from repeated experiments
    save_results(f'{current_data_dir}/experiment_repeats_results.pkl', {
        'all_results_dict': all_results_dict,
        'all_results_list': all_results_list
    })

    # Plot results from repeated experiments
    plot_experiment_results(max_grad_norms, all_results_list, all_results_dict, REPEATS, EPSILON, current_image_dir)

    # Run the final experiment
    train_losses, val_losses, clipped_proportions, train_means, val_means = run_single_optimized_experiment(
        max_grad_norms, optimized_params, train_dataset, val_dataset, epochs=EPOCHS, epsilon=EPSILON, input_dim=D
    )

    # Save final experiment results
    save_results(
        f'{current_data_dir}/final_experiment_results.pkl',
        (train_losses, val_losses, clipped_proportions, train_means, val_means),
    )

    # Plot final experiment results
    plot_clipped_proportions_per_iteration(max_grad_norms, clipped_proportions)

    plot_last_clipped_proportion_vs_mgn(
        max_grad_norms,
        clipped_proportions,
        save_fname=f'{current_image_dir}/toy-model-clipped-proportions-last-iteration-epsilon-{EPSILON}.png',
    )

    plot_predicted_means_all_epsilons(
        max_grad_norms,
        train_means,
        val_means,
        save_fname=f'{current_image_dir}/toy-model-norm-of-predicted-means-epsilon-{EPSILON}.png',
    )

    print(f'Finished experiment for seed = {seed}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment with specified seed.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for the experiment')
    args = parser.parse_args()

    main(data_dir='toy-model-data-latest', image_dir='temp-latest', seed=args.seed)
