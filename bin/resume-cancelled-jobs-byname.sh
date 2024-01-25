#!/bin/bash

if ! python -c "import optuna" &> /dev/null; then
    echo "Error: 'optuna' module not found. Please make sure you have correct environment activated."
    exit 1
fi

# Function to display usage/help
usage() {
    echo "Usage: $0 <EXPERIMENT_BASE> <MAX_TRIALS> <EXPERIMENT_NAME>"
    echo "Example: $0 /path/to/experiments 20"
}

# Check if the number of arguments is less than 3, display usage
if [ $# -lt 3 ]; then
    usage
    exit 1
fi

# Get command line arguments to variables
EXPERIMENT_BASE=$1
MAX_TRIALS=$2
EXPERIMENT_NAME=${3:-$LOG_PATTERN}

# Get the number of completed trials using the Python script
N_TRIALS=$(python bin/get_n_trials.py --optuna-journal "$EXPERIMENT_BASE"/data/optuna.journal --study-name "$EXPERIMENT_NAME")

# Calculate remaining trials
REMAINING_TRIALS=$((MAX_TRIALS - N_TRIALS))

# Resume the experiment
if [ "$REMAINING_TRIALS" -lt 0 ]; then
    REMAINING_TRIALS=0  # Just the final training round
fi

bash "$EXPERIMENT_BASE/scripts/resume.sh" "$EXPERIMENT_NAME" "$REMAINING_TRIALS"
