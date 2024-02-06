import glob
import ipdb
import json
import pandas as pd
import os

# Helper function to read JSON file
def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Aggregating trials data along with configuration details
def aggregate_trials_data(exp_dir):
    trials_file = os.path.join(exp_dir, 'trials.csv')
    config_file = os.path.join(exp_dir, 'configuration.json')
    hyperparams_file = os.path.join(exp_dir, 'hyperparameters.json')

    if os.path.exists(trials_file) and os.path.exists(config_file) and os.path.exists(hyperparams_file):
        df = pd.read_csv(trials_file, index_col=None, header=0)
        config = read_json(config_file)
        hyperparams = read_json(hyperparams_file)

        df['batch_size'] = hyperparams['batch_size']
        df['epsilon'] = hyperparams['target_epsilon']
        df['model_name'] = config['model_name']
        df['dataset_name'] = config['dataset_name']
        df['subset_size'] = config['subset_size']
        df['experiment_name'] = config['experiment_name']

        return df
    else:
        return pd.DataFrame()

def analyze_experiment_trials(df):
    best_trial_accuracy = df['value'].max()
    best_trial_params = df.loc[
        df['value'].idxmax(),
        [
            'params_learning_rate',
            'params_epochs',
            'params_max_grad_norm',
            'value',
        ]
    ]

    # Define thresholds for "close to best" and "distance from optimal learning rate"
    lr_threshold = 0.10
    accuracy_diff_threshold = 0.05

    close_lr_min = best_trial_params['params_learning_rate'] * (1 - lr_threshold)
    close_lr_max = best_trial_params['params_learning_rate'] * (1 + lr_threshold)

    filtered_df = df[
        (df['params_learning_rate'] >= close_lr_min)
        & (df['params_learning_rate'] <= close_lr_max)
        & (df['params_epochs'] > 1)].copy()

    filtered_df['accuracy_diff'] = abs(filtered_df['value'] - best_trial_accuracy)
    filtered_df['best_value'] = best_trial_accuracy
    filtered_df['best_learning_rate'] = best_trial_accuracy
    filtered_df['optimal_learning_rate'] = best_trial_params['params_learning_rate']
    filtered_df['optimal_epochs'] = best_trial_params['params_epochs']
    filtered_df['optimal_max_grad_norm'] = best_trial_params['params_max_grad_norm']

    bad_params_df = filtered_df[(filtered_df['value'] < best_trial_accuracy) & (filtered_df['accuracy_diff'] > accuracy_diff_threshold)]

    return bad_params_df

# Main execution
if __name__ == '__main__':
    root_dir = 'experiments/00-experiment-batch-size-variation/data'
    experiment_dirs = glob.glob(os.path.join(root_dir, '*'), recursive=True)
    bad_hypers = {}

    for exp_dir in experiment_dirs:
        df = aggregate_trials_data(exp_dir)
        if not df.empty:
            bad_params_df = analyze_experiment_trials(df)
            if not bad_params_df.empty:
                print('---------------------------------------------------------')
                print(f'Bad Params for {exp_dir.split(os.sep)[-1]}:')
                value, best_value = bad_params_df['value'].item(), bad_params_df['best_value'].item()
                print(f'Accuracy/Best accuracy: {value:.4f}/{best_value:.4f}')

                batch_size = bad_params_df['batch_size'].item()
                print(f'Batch size: {batch_size}')

                learning_rate, optimal_learning_rate = bad_params_df['params_learning_rate'].item(), bad_params_df['optimal_learning_rate'].item()
                print(f'Learning rate/Optimal learning rate: {learning_rate:.7f}/{optimal_learning_rate:.7f}')

                epochs, optimal_epochs = bad_params_df['params_epochs'].item(), bad_params_df['optimal_epochs'].item()
                print(f'Epochs/Optimal epochs: {epochs}/{optimal_epochs}')

                max_grad_norm, optimal_max_grad_norm = bad_params_df['params_max_grad_norm'].item(), bad_params_df['optimal_max_grad_norm'].item()
                print(f'Max grad norm/Optimal max grad norm: {max_grad_norm:.2f}/{optimal_max_grad_norm:.2f}')

                experiment_name = bad_params_df['experiment_name'].item()
                bad_hypers[experiment_name] = {
                    'accuracy': bad_params_df['value'].item(),
                    'best_accuracy': bad_params_df['best_value'].item(),
                    'batch_size': bad_params_df['batch_size'].item(),
                    'learning_rate': bad_params_df['params_learning_rate'].item(),
                    'optimal_learning_rate': bad_params_df['optimal_learning_rate'].item(),
                    'epochs': bad_params_df['params_epochs'].item(),
                    'optimal_epochs': bad_params_df['optimal_epochs'].item(),
                    'max_grad_norm': bad_params_df['params_max_grad_norm'].item(),
                    'optimal_max_grad_norm': bad_params_df['optimal_max_grad_norm'].item()
                }
    with open('bad_hypers.json', 'w') as f:
        print(f'Saving bad hypers to `bad_hypers.json`...')
        json.dump(bad_hypers, f, indent=4)

