#!/bin/bash

# Define the fixed argument for max time steps
T_max=50000000000 # i.e., 50 seconds

# Define the types of sampling available
sampling_types=("priority") # ("uniform" "priority")

# Define the gamma and base_reward pairs (to keep value of reward + gamma under 1)
gamma_base_pairs=("0.01 0.6" "0.5 0.24" "0.99 0.006") # ("0.01 0.6" "0.5 0.24" "0.99 0.006")

# Define the max_engagement_length values as an array
training_max_engagement_lengths=(100 75 50 25) # (50 25 5)
evaluation_max_engagement_lengths=(100 75 50 25)

# Calculate the total number of experiments
total_experiments=$((${#sampling_types[@]} * ${#gamma_base_pairs[@]} * ${#training_max_engagement_lengths[@]} \
                     * (${#evaluation_max_engagement_lengths[@]} + 1) ))
echo "Total number of experiments: $total_experiments"

# Initialize the experiment count
experiment_count=1

# Extra command line options for scenarios
scenario="split" # "split" will split the samples from train/eval directory based on split ratio, while "whole" will use all the samples in the specified train/eval directories
split_ratio=0.7 # 70% of the samples will be used for training and 30% for evaluation
train_path="/home/preet_derasari/Journal/datasets/RanSMAP/dataset/original"
eval_path="/home/preet_derasari/Journal/datasets/RanSMAP/dataset/original" # same as training since we are splitting the training dataset

# Base name of parent directory for the run in results/ which can change based on the train and eval paths above
train_name="original"
eval_name="original"

# Loop through the sampling types
for sampling_type in "${sampling_types[@]}"; do
    # Create the subdirectory for the sampling type
    results_dir="results/Train-${train_name}_Eval-${eval_name}/sampling-${sampling_type}"
    mkdir -p "$results_dir"

    # Loop through the gamma and base_reward pairs
    for pair in "${gamma_base_pairs[@]}"; do
        gamma=$(echo $pair | awk '{print $1}')
        base_reward=$(echo $pair | awk '{print $2}')

        # Create the subdirectory for the gamma and base_reward combination
        gamma_dir="$results_dir/gamma-${gamma}_base-${base_reward}"
        mkdir -p "$gamma_dir"

        # Inner loop for training mode
        for train_max_engagement_length in "${training_max_engagement_lengths[@]}"; do
            # Create the subdirectory for the training length
            train_dir="$gamma_dir/Train_len-${train_max_engagement_length}"
            mkdir -p "$train_dir"

            echo "Running experiment $experiment_count of $total_experiments with args:"
            echo "--T_max $T_max --mode "Training" --max_engagement_length $train_max_engagement_length \
            --gamma $gamma --base_reward $base_reward --results_dir "$train_dir" --scenario "$scenario" \
            --split_ratio $split_ratio --train_path "$train_path" --sampling_type "$sampling_type" "
            
            # Run the main script with the specified arguments for training
            python main.py --T_max $T_max --mode "Training" --max_engagement_length $train_max_engagement_length \
            --gamma $gamma --base_reward $base_reward --results_dir "$train_dir" --scenario "$scenario" \
            --split_ratio $split_ratio --train_path "$train_path" --sampling_type "$sampling_type"
            experiment_count=$((experiment_count+1))

            # Inner loop for evaluation mode
            for eval_max_engagement_length in "${evaluation_max_engagement_lengths[@]}"; do
                # Create the subdirectory for the evaluation length
                eval_dir="$train_dir/Eval_len-${eval_max_engagement_length}"
                mkdir -p "$eval_dir"

                echo "Running experiment $experiment_count of $total_experiments with args:"
                echo "--T_max $T_max --mode "Evaluation" --max_engagement_length $eval_max_engagement_length \
                --gamma $gamma --base_reward $base_reward --results_dir "$eval_dir" --scenario "$scenario" \
                --split_ratio $split_ratio --eval_path "$eval_path" --sampling_type "$sampling_type" "

                # Run the main script with the specified arguments for evaluation
                python main.py --T_max $T_max --mode "Evaluation" --max_engagement_length $eval_max_engagement_length \
                --gamma $gamma --base_reward $base_reward --results_dir "$eval_dir" --scenario "$scenario" \
                --split_ratio $split_ratio --eval_path "$eval_path" --sampling_type "$sampling_type"
                experiment_count=$((experiment_count+1))
            done
        done
    done
done