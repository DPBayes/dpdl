import json
import sys

def read_aggregated_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def format_float(value, precision=2):
    return f'{float(value):.{precision}f}'

def generate_markdown_tables(data):
    tables = {}

    for key, value in data.items():
        model_name = value['configuration']['model_name']
        dataset_name = value['configuration']['dataset_name']
        subset_size = value['configuration']['subset_size']
        epsilon = value['hyperparameters']['target_epsilon']
        batch_size = value['best_params']['batch_size']
        peft = value['configuration']['peft']
        n_trials = value['configuration']['n_trials']
        accuracy = format_float(value['best_value'])

        table_key = (dataset_name, model_name, f'{int(subset_size * 100)}%')
        if table_key not in tables:
            tables[table_key] = []

        tables[table_key].append({
            'peft': peft,
            'subset': f'{int(subset_size * 100)}%',
            'epsilon': epsilon,
            'batch_size': batch_size,
            'epochs': value['best_params']['epochs'],
            'learning_rate': format_float(value['best_params']['learning_rate'], 6),
            'max_grad_norm': format_float(value['best_params']['max_grad_norm']),
            'accuracy': accuracy
        })

    # Generate Markdown sorted by dataset, model, subset size, epsilon, and PEFT method
    markdown_output = ''
    for dataset in ['cifar10', 'cifar100']:
        for ((dataset_name, model, subset), rows) in sorted(tables.items()):
            if dataset_name == dataset:
                markdown_output += f'### {dataset.upper()} ({subset} Subset, {n_trials} Trials)\n\n#### {model}\n\n'
                markdown_output += '| PEFT Method | Epsilon | Optimized batch size | Optimized epochs | Optimized learning rate | Optimized max gradient norm | Accuracy |\n'
                markdown_output += '|-------------|---------|----------------------|------------------|-------------------------|-----------------------------|----------|\n'

                for row in sorted(rows, key=lambda x: (x['epsilon'], x['peft'])):
                    markdown_output += f"| {row['peft']} | {row['epsilon']} | {row['batch_size']} | {row['epochs']} | {row['learning_rate']} | {row['max_grad_norm']} | {row['accuracy']} |\n"

                markdown_output += '\n'

    return markdown_output

def main():
    if len(sys.argv) != 2:
        print('Usage: python script.py <path_to_aggregated_data_json>')
        sys.exit(1)

    data_file = sys.argv[1]
    data = read_aggregated_data(data_file)

    print(generate_markdown_tables(data))

if __name__ == '__main__':
    main()
