import os
import json
import sys

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

def process_experiment_directory(directory):
    data = {}

    print(f'Processing experiments in directory: {directory}')

    for entry in os.listdir(directory):
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
    if len(sys.argv) < 2:
        print('Usage: python script.py <directory1> [<directory2> ...]')
        sys.exit(1)

    all_data = {}
    for directory in sys.argv[1:]:
        all_data.update(process_experiment_directory(directory))

    with open('aggregated_data.json', 'w') as outfile:
        json.dump(all_data, outfile, indent=4)

if __name__ == '__main__':
    main()
