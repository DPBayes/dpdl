#!/bin/bash

set -euo pipefail

if ! python -c "import optuna" &> /dev/null; then
    echo "Error: 'optuna' module not found. Please make sure you have correct environment activated."
    exit 1
fi

MAX_TRIALS=${1:-20}

for file in slurm-*.out; do
    # Check if the jobs was cancelled due to time out
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

        # Check if a seed is present in the experiment name and adjust experiment_base
        if [[ $experiment_name =~ Seed([0-9]+) ]]; then
            seed=${BASH_REMATCH[1]}
            experiment_base="${experiment_base}__Extension_Seed${seed}"
        fi

        # Check that the job hasn't been completed
        if [ ! -f "experiments/$experiment_base/$experiment_name/data/runtime" ]; then
            # Check if the experiment isin Slurm queue
            if squeue --me -o "%.150j" | grep -q "$experiment_name"; then
                echo "Experiment $experiment_name is already in queue."
            else
                echo "Job for experiment $experiment_name is not in queue, resuming.."

                # Get the number of completed trials using the Python script
                n_trials=$(python bin/get_n_trials.py --optuna-journal experiments/"$experiment_base"/data/optuna.journal --study-name "$experiment_name")

                re='^[0-9]+$'
                if ! [[ $n_trials =~ $re ]]; then
                    echo $n_trials
                else
                    # Calculate remaining trials
                    remaining_trials=$((MAX_TRIALS - n_trials))

                    # Resume the experiment
                    if [ "$remaining_trials" -lt 0 ]; then
                        remaining_trials = 0  # Just the final training round
                    fi

                    echo "Running:" "experiments/$experiment_base/scripts/resume.sh" "$experiment_name" "$remaining_trials"
                    bash "experiments/$experiment_base/scripts/resume.sh" "$experiment_name" "$remaining_trials"
                    echo " -> Experiment resumed."
                fi

            fi
        else
            echo "Runtime does not exist for $experiment_name!"
        fi
    fi
done
