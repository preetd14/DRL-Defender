#!/bin/bash

# Define the fixed argument
T_d=50000000000

# Define the gamma and base_reward pairs (to keep value of reward + gamma under 1)
gamma_base_pairs=("0.01 0.6" "0.99 0.006")

# Calculate the total number of experiments
total_experiments=$(( ${#gamma_base_pairs[@]} * 2 * 3 * 3 ))
experiment_count=1

# Loop through the gamma and base_reward pairs
for pair in "${gamma_base_pairs[@]}"; do
    gamma=$(echo $pair | awk '{print $1}')
    base_reward=$(echo $pair | awk '{print $2}')

    # Create the subdirectory for the gamma and base_reward combination
    results_dir="results/Train-original_Eval-extra/gamma-${gamma}_base-${base_reward}"
    mkdir -p "$results_dir"

    # Inner loop for training mode
    for max_engagement_length in 50 25 5; do
        # Create the subdirectory for the training length
        train_dir="$results_dir/Train_len-${max_engagement_length}"
        mkdir -p "$train_dir"

        echo "Running experiment with args: --T_d $T_d --mode "Training" --max_engagement_length $max_engagement_length --gamma $gamma --base_reward $base_reward --results_dir "$train_dir" "
        echo "$experiment_count of $total_experiments"
        
        # Run the main script with the specified arguments for training
        python main.py --T_d $T_d --mode "Training" --max_engagement_length $max_engagement_length --gamma $gamma --base_reward $base_reward --results_dir "$train_dir"
        experiment_count=$((experiment_count+1))

        # Inner loop for evaluation mode
        for eval_max_engagement_length in 50 25 5; do
            # Create the subdirectory for the evaluation length
            eval_dir="$train_dir/Eval_len-${eval_max_engagement_length}"
            mkdir -p "$eval_dir"

            echo "Running experiment with args: --T_d $T_d --mode "Training" --max_engagement_length $max_engagement_length --gamma $gamma --base_reward $base_reward --results_dir "$train_dir" "
            echo "$experiment_count of $total_experiments"

            # Run the main script with the specified arguments for evaluation
            python main.py --T_d $T_d --mode "Evaluation" --max_engagement_length $eval_max_engagement_length --gamma $gamma --base_reward $base_reward --results_dir "$eval_dir"
            experiment_count=$((experiment_count+1))

        done
    done
done