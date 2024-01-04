import json
import sys

def read_experiment_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def format_float(value, precision=2):
    return f'{float(value):.{precision}f}'

def generate_markdown_tables(data):
    tables = {}

    for experiment_id, experiment_data in data.items():
        # Extracting model name from the 'configuration' section
        model_name = experiment_data['configuration']['model_name']
        dataset_name = experiment_data['configuration']['dataset_name']
        subset_size = experiment_data['configuration']['subset_size']
        epsilon = experiment_data['hyperparameters']['target_epsilon']
        batch_size = experiment_data['hyperparameters']['batch_size']
        batch_size = batch_size if batch_size != -1 else "Full batch"

        # Use a tuple of model, dataset, subset size, and epsilon as a key for tables
        table_key = (model_name, dataset_name, subset_size, epsilon)
        if table_key not in tables:
            tables[table_key] = []

        # Add experiment data to the table entry
        tables[table_key].append({
            'batch_size': batch_size,
            'epochs': experiment_data['best_params']['epochs'],
            'learning_rate': format_float(experiment_data['best_params']['learning_rate'], 6),
            'max_grad_norm': format_float(experiment_data['best_params']['max_grad_norm']),
            'accuracy': format_float(experiment_data['best_value'], 2)
        })

    # Generate Markdown sorted by model, dataset, subset size, and epsilon
    markdown_output = ''
    for (model, dataset, subset, epsilon) in sorted(tables.keys()):
        rows = tables[(model, dataset, subset, epsilon)]
        markdown_output += f'#### {model} on {dataset} ({subset*100}% Subset) - Epsilon {epsilon}\n\n'
        markdown_output += '| Batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |\n'
        markdown_output += '|------------|------------------|-------------------------|-----------------------------|----------|\n'

        # Sorting rows to ensure "Full batch" appears last
        sorted_rows = sorted(rows, key=lambda x: (x['batch_size'] == 'Full batch', x['batch_size']))
        for row in sorted_rows:
            markdown_output += f"| {row['batch_size']} | {row['epochs']} | {row['learning_rate']} | {row['max_grad_norm']} | {row['accuracy']} |\n"

        markdown_output += '\n'

    return markdown_output

def main():
    if len(sys.argv) != 2:
        print('Usage: python script.py <path_to_experiment_data_json>')
        sys.exit(1)

    data_file = sys.argv[1]
    data = read_experiment_data(data_file)

    print(generate_markdown_tables(data))

if __name__ == '__main__':
    main()
