import numpy as np
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


def main_load_and_plot(data_dir='toy-model-data', image_dir='temp'):
    setup_directories(data_dir, image_dir)

    # Load optimized hyperparameters
    print('Loading optimized hyperparameters from file...')
    optimized_params = load_results(f'{data_dir}/optimized_params.pkl')

    # Load repeated experiment results
    print('Loading experiment repeat results from file...')
    repeat_results = load_results(f'{data_dir}/experiment_repeats_results.pkl')
    all_results_dict, all_results_list = repeat_results.values()

    # Load final experiment results
    print('Loading final experiment results from file...')
    (
        train_losses,
        val_losses,
        clipped_proportions,
        train_means,
        val_means,
    ) = load_results(f'{data_dir}/final_experiment_results.pkl')

    # Experiment parameters
    REPEATS = 200
    EPSILON = 0.25
    MAX_GRAD_NORMS = np.geomspace(1e-4, 30, 15)

    # Plot results from repeated experiments
    print('Plotting results from repeated experiments...')
    plot_all_repeats(
        MAX_GRAD_NORMS,
        all_results_list,
        REPEATS,
        EPSILON,
        save_fname=f'{image_dir}/toy-model-repeats-{REPEATS}-with-optimized-hypers-epsilon-{EPSILON}.png',
    )
    plot_mean_with_confidence_intervals(
        MAX_GRAD_NORMS,
        all_results_dict,
        REPEATS,
        EPSILON,
        save_fname=f'{image_dir}/toy-model-repeats-{REPEATS}-with-optimized-hypers-epsilon-{EPSILON}-mean-with-confidence.png',
    )

    # Plot final experiment results
    print('Plotting final experiment results...')
    plot_clipped_proportions_per_iteration(MAX_GRAD_NORMS, clipped_proportions)
    plot_last_clipped_proportion_vs_mgn(
        MAX_GRAD_NORMS,
        clipped_proportions,
        save_fname=f'{image_dir}/toy-model-clipped-proportions-last-iteration-epsilon-{EPSILON}.png',
    )
    plot_predicted_means_all_epsilons(
        MAX_GRAD_NORMS,
        train_means,
        val_means,
        save_fname=f'{image_dir}/toy-model-norm-of-predicted-means-epsilon-{EPSILON}.png',
    )

    print('All plots generated successfully.')


def main(data_dir='toy-model-data', image_dir='temp'):
    # Initialize seeds, directories, and other configuration
    seed = 42
    seed_everything(seed)
    DATA_DIR = data_dir
    IMAGE_DIR = image_dir
    setup_directories(DATA_DIR, IMAGE_DIR)

    # Experiment parameters
    D = 10  # Dimensionality of the data
    N = 1250  # Number of data points
    EPOCHS = 10
    REPEATS = 2
    EPSILON = 0.25
    MAX_GRAD_NORMS = np.geomspace(1e-4, 30, 15)

    # Generate dataset
    data, target = generate_mixture_data(N, D)
    dataset = TensorDataset(data, target)

    # Split the dataset once and reuse the splits across experiments
    train_size = int(0.8 * N)
    val_size = N - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Step 1: Optimize hyperparameters for each max_grad_norm
    optimized_losses = []
    optimized_params = []
    for max_grad_norm in MAX_GRAD_NORMS:
        print(f'Optimizing hyperparameters for max_grad_norm = {max_grad_norm}')
        best_loss, best_params = optimize_hyperparameters(
            train_dataset,
            val_dataset,
            max_grad_norm,
            n_trials=2,
            epochs=EPOCHS,
            epsilon=EPSILON,
            input_dim=D,
            seed=seed,
        )
        optimized_losses.append(best_loss)
        optimized_params.append(best_params)

    # Save optimized parameters
    save_results(f'{DATA_DIR}/optimized_params.pkl', optimized_params)

    # Run repeated experiments with optimized hyperparameters
    all_results_dict, all_results_list = run_repeats_with_optimized_hypers(
        MAX_GRAD_NORMS,
        optimized_params,
        train_dataset,
        val_dataset,
        repeats=REPEATS,
        epochs=EPOCHS,
        epsilon=EPSILON,
        input_dim=D,
    )

    # Save results from repeated experiments
    save_results(
        f'{DATA_DIR}/experiment_repeats_results.pkl',
        {
            'all_results_dict': all_results_dict,
            'all_results_list': all_results_list,
        },
    )

    # Plot results from repeated experiments
    plot_all_repeats(
        MAX_GRAD_NORMS,
        all_results_list,
        REPEATS,
        EPSILON,
        save_fname=f'{IMAGE_DIR}/toy-model-repeats-{REPEATS}-with-optimized-hypers-epsilon-{EPSILON}.png',
    )

    plot_mean_with_confidence_intervals(
        MAX_GRAD_NORMS,
        all_results_dict,
        REPEATS,
        EPSILON,
        save_fname=f'{IMAGE_DIR}/toy-model-repeats-{REPEATS}-with-optimized-hypers-epsilon-{EPSILON}-mean-with-confidence.png',
    )

    # Run the final experiment
    (
        train_losses,
        val_losses,
        clipped_proportions,
        train_means,
        val_means,
    ) = run_single_optimized_experiment(
        MAX_GRAD_NORMS,
        optimized_params,
        train_dataset,
        val_dataset,
        epochs=EPOCHS,
        epsilon=EPSILON,
        input_dim=D,
    )

    # Save final experiment results
    save_results(
        f'{DATA_DIR}/final_experiment_results.pkl',
        (train_losses, val_losses, clipped_proportions, train_means, val_means),
    )

    # Plot results
    plot_clipped_proportions_per_iteration(MAX_GRAD_NORMS, clipped_proportions)

    plot_last_clipped_proportion_vs_mgn(
        MAX_GRAD_NORMS,
        clipped_proportions,
        save_fname=f'{IMAGE_DIR}/toy-model-clipped-propertions-last-iteration-epsilon-{EPSILON}.png',
    )

    plot_predicted_means_all_epsilons(
        MAX_GRAD_NORMS,
        train_means,
        val_means,
        save_fname=f'{IMAGE_DIR}/toy-model-norm-of-predicted-means-epsilon-{EPSILON}.png',
    )


if __name__ == '__main__':
    #main_load_and_plot('toy-model-data', 'temp')
    main('toy-model-data', 'temp')
