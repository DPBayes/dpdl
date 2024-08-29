import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_clipped_proportions_per_iteration(max_grad_norms, clipped_proportions_dict):
    cmap = plt.colormaps['tab20']

    for i, epsilon in enumerate(clipped_proportions_dict.keys()):
        plt.figure(figsize=(10, 6))

        for idx, mgn in enumerate(max_grad_norms):
            clipped_proportions = clipped_proportions_dict[epsilon][idx]
            plt.plot(
                clipped_proportions, color=cmap(idx % cmap.N), label=f'MGN = {mgn:.4f}'
            )

        plt.xlabel('Iteration')
        plt.ylabel('Proportion of Clipped Gradients')
        plt.title(
            f'Proportion of clipped gradients vs. Iteration (Epsilon = {epsilon})'
        )
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f'temp/proportion-of-clipped-gradients-epsilon{epsilon}.png')
        ##plt.show()


def plot_last_clipped_proportion_vs_mgn(
    max_grad_norms,
    clipped_proportions_dict,
    save_fname=None,
    log_scale=True,
):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(clipped_proportions_dict.keys())))

    for i, (epsilon, proportions_list) in enumerate(clipped_proportions_dict.items()):
        last_clipped_proportions = [proportions[-1] for proportions in proportions_list]
        plt.plot(
            max_grad_norms,
            last_clipped_proportions,
            color=colors[i],
            marker='o',
            label=f'Epsilon = {epsilon}',
        )

    if log_scale:
        plt.xscale('log')

    plt.xlabel('Max Gradient Norm (Log Scale)')
    plt.ylabel('Proportion of clipped gradients (Last Iteration)')
    plt.title('Proportion of clipped Gradients vs. Max Gradient Norm')
    plt.grid(True)
    plt.legend(loc='upper right')

    if save_fname:
        plt.savefig(save_fname)

    #plt.show()


def plot_predicted_means_all_epsilons(
    max_grad_norms,
    predicted_means_train_dict,
    predicted_means_val_dict,
    save_fname=None,
    log_scale=True,
):
    epsilons = list(predicted_means_train_dict.keys())
    palette = sns.color_palette('tab10', len(epsilons))

    plt.figure(figsize=(10, 6))

    for idx, epsilon in enumerate(epsilons):
        color = palette[idx]
        final_train_means = [
            predicted_means_train_dict[epsilon][i] for i in range(len(max_grad_norms))
        ]
        final_val_means = [
            predicted_means_val_dict[epsilon][i] for i in range(len(max_grad_norms))
        ]

        plt.plot(
            max_grad_norms,
            final_train_means,
            label=f'Training, Epsilon={epsilon}',
            marker='o',
            linestyle='--',
            color=color,
        )
        plt.plot(
            max_grad_norms,
            final_val_means,
            label=f'Validation, Epsilon={epsilon}',
            marker='x',
            linestyle='-',
            color=color,
        )

    if log_scale:
        plt.xscale('log')

    plt.xlabel('Max Gradient Norm')
    plt.ylabel('Last Iteration Predicted Mean Norm')
    plt.title(f'Last Iteration Predicted Mean Norms Across All Epsilons')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.tight_layout()

    if save_fname:
        plt.savefig(save_fname)

    #plt.show()


def plot_mean_with_confidence_intervals(
    max_grad_norms,
    all_results_dict,
    n_repeats,
    epsilon,
    save_fname=None,
    log_scale=True,
):
    means = []
    conf_intervals = []

    for mgn in max_grad_norms:
        losses = all_results_dict[mgn]
        mean_loss = np.mean(losses)
        std_dev = np.std(losses)
        conf_interval = 1.96 * (std_dev / np.sqrt(len(losses)))

        means.append(mean_loss)
        conf_intervals.append(conf_interval)

    plt.figure(figsize=(12, 8))
    plt.plot(max_grad_norms, means, color='b', marker='o', label='Mean MSE Loss')
    plt.fill_between(
        max_grad_norms,
        np.array(means) - np.array(conf_intervals),
        np.array(means) + np.array(conf_intervals),
        color='b',
        alpha=0.2,
        label='95% CI',
    )

    if log_scale:
        plt.xscale('log')

    # Set the ticks where the observations are, rounded to 2 decimals and rotated 45 degrees
    plt.xticks(ticks=max_grad_norms, labels=[f'{x:.2f}' for x in max_grad_norms], rotation=45)

    plt.xlabel('Maximum Gradient Norm')
    plt.ylabel('Mean MSE Loss')
    plt.title(
        f'Mean MSE Loss with 95% Confidence Intervals ({n_repeats} repeats) (ε = {epsilon})'
    )
    plt.legend()
    plt.grid(True)
    #plt.ylim([1.9, 2.1])

    if save_fname:
        plt.savefig(save_fname)

    # plt.show()


def plot_all_repeats(max_grad_norms, all_results, n_repeats, epsilon, save_fname=None, log_scale=True):
    plt.figure(figsize=(12, 8))

    for i, repeat_losses in enumerate(all_results):
        plt.plot(max_grad_norms, repeat_losses, alpha=0.5, color='blue', linewidth=0.5)

    if log_scale:
        plt.xscale('log')

    plt.xlabel('Maximum Gradient Norm')
    plt.ylabel('MSE Loss')
    plt.title(f'All {n_repeats} Repeats for MSE Losses (ε = {epsilon})')
    plt.grid(True)

    if save_fname:
        plt.savefig(save_fname)

    #plt.show()
