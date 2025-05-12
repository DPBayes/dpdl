import argparse
import os
import json
import sys
import re
import csv

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

def process_experiment_directory(directory, pattern=None):
    data = []

    print(f'Processing experiments in directory: {directory}')
    for entry in os.listdir(directory):
        if pattern and not re.match(pattern, entry):
            continue

        experiment_path = os.path.join(directory, entry)

        if os.path.isdir(experiment_path):
            if not os.path.exists(os.path.join(experiment_path, 'runtime')):
                print(f'Warning: Experiment {entry} not finished. Skipping.')
                continue

            try:
                experiment_data = {
                    'hyperparameters': read_json_file(os.path.join(experiment_path, 'hyperparameters.json')),
                    'configuration': read_json_file(os.path.join(experiment_path, 'configuration.json')),
                    'gpu_count': read_text_file(os.path.join(experiment_path, 'gpu_count')),
                    'gpu_type': read_text_file(os.path.join(experiment_path, 'gpu_type')),
                    'runtime': read_text_file(os.path.join(experiment_path, 'runtime')),
                }

                git_hash_file_path = os.path.join(experiment_path, 'git-hash')
                if os.path.exists(git_hash_file_path):
                    experiment_data['git_hash'] = read_text_file(git_hash_file_path)

                best_value_file_path = os.path.join(experiment_path, 'best-value')
                if os.path.exists(best_value_file_path):
                    experiment_data['best_value'] = read_text_file(best_value_file_path)

                best_params_file_path = os.path.join(experiment_path, 'best-params.json')
                if os.path.exists(best_params_file_path):
                    experiment_data['best_params'] = read_json_file(best_params_file_path)

                final_metrics_file_path = os.path.join(experiment_path, 'final-metrics')
                if os.path.exists(final_metrics_file_path):
                    experiment_data['final_metrics'] = read_json_file(final_metrics_file_path)

                test_metrics_file_path = os.path.join(experiment_path, 'test_metrics')
                if os.path.exists(test_metrics_file_path):
                    test_metrics = read_json_file(test_metrics_file_path)

                    if 'ConfusionMatrix' in test_metrics:
                        del test_metrics['ConfusionMatrix']

                    experiment_data['test_metrics'] = test_metrics

                epoch_losses_file_path = os.path.join(experiment_path, 'epoch_losses.csv')
                if os.path.exists(epoch_losses_file_path):
                    experiment_data['losses_by_epoch'] = read_csv_file(epoch_losses_file_path)

                data.append(experiment_data)

            except Exception as e:
                raise Exception(f'Error processing {experiment_path}: {e}')
    return data

def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment data.')
    parser.add_argument('directories', nargs='+', help='Directories to process')
    parser.add_argument('-f', '--filter', help='Regex pattern to filter directories', default=None)
    parser.add_argument('-o', '--output', help='Output file name', default='aggregated_data.json')

    args = parser.parse_args()

    all_data = []
    for directory in args.directories:
        all_data.extend(process_experiment_directory(directory, args.filter))

    with open(args.output, 'w') as outfile:
        json.dump(all_data, outfile, indent=4)

if __name__ == '__main__':
    main()
