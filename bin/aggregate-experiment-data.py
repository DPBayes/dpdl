import os
import json
import sys

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def process_experiment_directory(parent_directory, filter_str=None):
    all_data = {}

    for directory in os.listdir(parent_directory):
        if filter_str and filter_str not in directory:
            continue

        print(f'Processing experiment: {directory}')

        experiment_path = os.path.join(parent_directory, directory)
        if os.path.isdir(experiment_path):
            try:
                data = {
                    'best_params': read_json_file(os.path.join(experiment_path, 'best-params.json')),
                    'best_value': read_text_file(os.path.join(experiment_path, 'best-value')),
                    'hyperparameters': read_json_file(os.path.join(experiment_path, 'hyperparameters.json')),
                    'configuration': read_json_file(os.path.join(experiment_path, 'configuration.json')),
                    'gpu_count': read_text_file(os.path.join(experiment_path, 'gpu_count')),
                    'gpu_type': read_text_file(os.path.join(experiment_path, 'gpu_type')),
                    'runtime': read_text_file(os.path.join(experiment_path, 'runtime')),
                    'git_hash': read_text_file(os.path.join(experiment_path, 'git-hash'))
                }
                all_data[directory] = data
            except Exception as e:
                raise Exception(f'Error processing {experiment_path}: {e}')

    return all_data

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python script.py <parent_directory> [filter_str]')
        sys.exit(1)

    parent_directory = sys.argv[1]
    filter_str = sys.argv[2] if len(sys.argv) == 3 else None
    combined_data = process_experiment_directory(parent_directory, filter_str)

    with open('aggregated_data.json', 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)

if __name__ == '__main__':
    main()
