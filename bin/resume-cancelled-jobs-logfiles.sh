#!/bin/bash

if ! python -c "import optuna" &> /dev/null; then
    echo "Error: 'optuna' module not found. Please make sure you have correct environment activated."
    exit 1
fi

MAX_TRIALS=${1:-20}

for file in slurm-*.out; do
    # Check if the file contains the text "CANCELLED"
    if grep -q "JOB.*CANCELLED.*DUE TO TIME LIMIT" "$file"; then
        # Extract the experiment name
        experiment_name=$(echo "$file" | sed -E 's/slurm-(.*)\.[0-9]+\.out/\1/')

        # Define the base path for different experiment types
        if [[ $experiment_name == *"LearningRate"* ]]; then
            experiment_base="03-learning-rate-variation"
        elif [[ $experiment_name == *"MaxGradNorm"* ]]; then
            experiment_base="02-maximum-gradient-norm-variation"
        elif [[ $experiment_name == *"Epoch"* ]]; then
            experiment_base="04-epoch-variation"
        else
            continue
        fi

        # Check for the existence of the runtime file
        if [ ! -f "experiments/$experiment_base/$experiment_name/data/runtime" ]; then
            # Check if the experiment is running in Slurm
            if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
                echo "Experiment $experiment_name is already in queue."
            else
                echo "Job for experiment $experiment_name is not in queue, resuming.."

                # Get the number of completed trials using the Python script
                n_trials=$(python bin/get_n_trials.py --optuna-journal experiments/"$experiment_base"/data/optuna.journal --study-name "$experiment_name")

                # Calculate remaining trials
                remaining_trials=$((MAX_TRIALS - n_trials))

                # Resume the experiment
                if [ "$remaining_trials" -gt -1 ]; then
                    echo "Running:" "experiments/$experiment_base/scripts/resume.sh" "$experiment_name" "$remaining_trials"
                    #bash "experiments/$experiment_base/scripts/resume.sh" "$experiment_name" "$remaining_trials"
                    echo " -> Experiment resumed."
                fi

            fi
        else
            echo "Runtime does not exist for $experiment_name!"
        fi
    fi
done
