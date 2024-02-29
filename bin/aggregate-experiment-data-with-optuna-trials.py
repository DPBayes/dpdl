import argparse
import os
import json
import sys
import re
import csv
import optuna

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def read_csv_file(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def get_optuna_storage(optuna_journal_fname):
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(optuna_journal_fname),
    )

    return storage

def get_best_trial_value_from_journal(journal_path, experiment_name, trial_number=None):
    # Load the Optuna storage
    storage = get_optuna_storage(journal_path)
    study = optuna.load_study(study_name=experiment_name, storage=storage)

    # If a trial number is specified, filter trials up to that number
    if trial_number is not None:
        trials = [trial for trial in study.trials if trial.value is not None and trial.number <= trial_number]

        if len(trials) < trial_number:
            print(f'Warning: Experiment {experiment_name} has only {len(trials)} trials an {trial_number} is requested.')

        best_trial = max(trials, key=lambda trial: trial.value)
    else:
        best_trial = study.best_trial

    return best_trial.value

def process_experiment_directory(directory, pattern=None, trial_number=None):
    data = {}
    journal_path = os.path.join(directory, 'optuna.journal')

    print(f'Processing experiments in directory: {directory}')
    for entry in os.listdir(directory):
        if pattern and not re.match(pattern, entry):
            continue

        experiment_path = os.path.join(directory, entry)
        if os.path.isdir(experiment_path):
            try:
                experiment_data = {
                    'hyperparameters': read_json_file(os.path.join(experiment_path, 'hyperparameters.json')),
                    'configuration': read_json_file(os.path.join(experiment_path, 'configuration.json')),
                }

                runtime_file_path = os.path.join(experiment_path, 'runtime')
                if os.path.exists(runtime_file_path):
                    experiment_data['gpu_type'] = read_text_file(runtime_file_path)

                gpu_type_file_path = os.path.join(experiment_path, 'gpu_type')
                if os.path.exists(gpu_type_file_path):
                    experiment_data['gpu_type'] = read_text_file(gpu_type_file_path)

                gpu_count_file_path = os.path.join(experiment_path, 'gpu_count')
                if os.path.exists(gpu_count_file_path):
                    experiment_data['gpu_count'] = read_text_file(gpu_count_file_path)

                git_hash_file_path = os.path.join(experiment_path, 'git-hash')
                if os.path.exists(git_hash_file_path):
                    experiment_data['git_hash'] = read_text_file(git_hash_file_path)

                best_value_file_path = os.path.join(experiment_path, 'best-value')
                if os.path.exists(best_value_file_path):
                    experiment_data['best_value'] = read_text_file(best_value_file_path)

                best_params_file_path = os.path.join(experiment_path, 'best-params.json')
                if os.path.exists(best_params_file_path):
                    experiment_data['best_params'] = read_json_file(best_params_file_path)

                snr_file_path = os.path.join(experiment_path, 'signal-to-noise-ratio.csv')
                if os.path.exists(snr_file_path):
                    experiment_data['signal_to_noise_ratio'] = read_csv_file(snr_file_path)

                if os.path.exists(journal_path):
                    best_value = get_best_trial_value_from_journal(journal_path, entry, trial_number)
                    experiment_data['best_trial_value'] = best_value

                data[entry] = experiment_data

            except Exception as e:
                raise Exception(f'Error processing {experiment_path}: {e}')
    return data

def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment data.')
    parser.add_argument('directories', nargs='+', help='Directories to process')
    parser.add_argument('-f', '--filter', help='Regex pattern to filter directories', default=None)
    parser.add_argument('-o', '--output', help='Output file name', default='aggregated_data_with_optuna_trials.json')
    parser.add_argument('-t', '--trial_number', type=int, help='Trial number to limit the search for the best trial', default=None)

    args = parser.parse_args()

    all_data = {}
    for directory in args.directories:
        all_data.update(process_experiment_directory(directory, args.filter, args.trial_number))

    with open(args.output, 'w') as outfile:
        json.dump(all_data, outfile, indent=4)

if __name__ == '__main__':
    main()
