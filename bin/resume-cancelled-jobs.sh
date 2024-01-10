#!/bin/bash

# Default log file pattern
LOG_PATTERN="slurm-*.out"

# Function to display usage/help
usage() {
    echo "Usage: $0 <EXPERIMENT_BASE> <MAX_TRIALS> [LOG_FILE_PATTERN]"
    echo "Example: $0 /path/to/experiments 20"
}

# Check if the number of arguments is less than 2, display usage
if [ $# -lt 2 ]; then
    usage
    exit 1
fi

# Get command line arguments to variables
EXPERIMENT_BASE=$1
MAX_TRIALS=$2
LOG_PATTERN=${3:-$LOG_PATTERN}

# List and process each log file
ls $LOG_PATTERN | while read -r file; do
    if grep -q "CANCELLED" "$file"; then
        # Extract experiment name from the file name
        EXPERIMENT_NAME=$(echo "$file" | grep -oP 'slurm-\K(.*)(?=\.out)' | sed 's/\.[0-9]*$//')

        # Get the number of completed trials using the Python script
        N_TRIALS=$(python bin/get_n_trials.py --optuna-journal "$EXPERIMENT_BASE"/data/optuna.journal --study-name "$EXPERIMENT_NAME")

        # Calculate remaining trials
        REMAINING_TRIALS=$((MAX_TRIALS - N_TRIALS))

        # Resume the experiment
        if [ "$REMAINING_TRIALS" -gt -1 ]; then
            bash "$EXPERIMENT_BASE/scripts/resume.sh" "$EXPERIMENT_NAME" "$REMAINING_TRIALS"
        fi
    fi
done
