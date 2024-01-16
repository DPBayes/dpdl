import argparse
import os
import json
import sys
import re

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def process_experiment_directory(directory, pattern=None):

    data = {}

    print(f'Processing experiments in directory: {directory}')
    for entry in os.listdir(directory):
        # Skip the entry if it does not match the pattern
        if pattern and not re.match(pattern, entry):
            continue

        experiment_path = os.path.join(directory, entry)
        if os.path.isdir(experiment_path):
            try:
                experiment_data = {
                    'best_params': read_json_file(os.path.join(experiment_path, 'best-params.json')),
                    'best_value': read_text_file(os.path.join(experiment_path, 'best-value')),
                    'hyperparameters': read_json_file(os.path.join(experiment_path, 'hyperparameters.json')),
                    'configuration': read_json_file(os.path.join(experiment_path, 'configuration.json')),
                    'gpu_count': read_text_file(os.path.join(experiment_path, 'gpu_count')),
                    'gpu_type': read_text_file(os.path.join(experiment_path, 'gpu_type')),
                    'runtime': read_text_file(os.path.join(experiment_path, 'runtime')),
                    'git_hash': read_text_file(os.path.join(experiment_path, 'git-hash'))
                }
                data[entry] = experiment_data
            except Exception as e:
                raise Exception(f'Error processing {experiment_path}: {e}')
    return data


def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment data.')
    parser.add_argument('directories', nargs='+', help='Directories to process')
    parser.add_argument('-f', '--filter', help='Regex pattern to filter directories', default=None)
    parser.add_argument('-o', '--output', help='Output file name', default='aggregated_data.json')

    args = parser.parse_args()

    all_data = {}
    for directory in args.directories:
        all_data.update(process_experiment_directory(directory, args.filter))

    with open(args.output, 'w') as outfile:
        json.dump(all_data, outfile, indent=4)

if __name__ == '__main__':
    main()
