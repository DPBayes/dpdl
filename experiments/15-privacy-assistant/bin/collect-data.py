import os
import json
import csv

def extract_metrics(subdir_path):
    """
    Extract epsilon and accuracy from experiment directory
    """
    final_metrics_path = os.path.join(subdir_path, 'final-metrics')

    if not os.path.exists(final_metrics_path):
        print(f'File "final-metrics" does not exist in {subdir_path}')
        return None, None

    with open(final_metrics_path, 'r') as f:
        final_metrics = json.load(f)
        accuracy = final_metrics.get('MulticlassAccuracy', None)

    hyperparameters_path = os.path.join(subdir_path, 'hyperparameters.json')
    with open(hyperparameters_path, 'r') as f:
        hyperparameters = json.load(f)
        epsilon = hyperparameters.get('target_epsilon', None)

    return epsilon, accuracy

def extract_configuration(subdir_path):
    """
    Extract dataset_name and subset_size from experiment configuration
    """
    configuration_path = os.path.join(subdir_path, 'configuration.json')

    if not os.path.exists(configuration_path):
        print(f'File "configuration.json" does not exist in {subdir_path}')
        return None, None

    with open(configuration_path, 'r') as f:
        configuration = json.load(f)
        dataset_name = configuration.get('dataset_name', None)
        subset_size = configuration.get('subset_size', None)

    return dataset_name, subset_size

def collect_data(base_dir):
    """
    Collect epsilon, accuracy, dataset_name, and subset_size for all experiment
    """
    data = []

    # Loop through experiment directories
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)

        if os.path.isdir(subdir_path) and 'Epsilon' in subdir:
            epsilon, accuracy = extract_metrics(subdir_path)
            dataset_name, subset_size = extract_configuration(subdir_path)

            data.append([epsilon, accuracy, dataset_name, subset_size])

    return data

def save_to_csv(data, output_file):
    """
    Write collected data to CSV
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epsilon', 'accuracy', 'dataset_name', 'subset_size'])
        writer.writerows(data)

def main():
    base_dir = '../data'
    output_file = f'{base_dir}/epsilon_accuracy_dataset.csv'

    data = collect_data(base_dir)

    # Save to CSV
    save_to_csv(data, output_file)
    print(f'Data successfully saved to {output_file}')

if __name__ == '__main__':
    main()

