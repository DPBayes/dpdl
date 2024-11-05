import argparse
import yaml

# input file path
parser = argparse.ArgumentParser(description="Read file path")
parser.add_argument("file_path", type=str, help="Path to the file to read")

# parse the arguments
args = parser.parse_args()

with open(args.file_path, "r") as file:
    config = yaml.safe_load(file)

for key, value in config.items():
    print(f"{key}: {value}")
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            print(f"  {subkey}: {subvalue}, type: {type(subvalue)}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            print(f"  {i}: {item}, type: {type(item)}")

