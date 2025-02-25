from agent import Agent
import argparse
import yaml
import os
import random
from scenario import Scenario
from results import compute_results

def main(args):
    """
    Main function to execute the DRL-based ransomware attack detection and defense system.

    This function orchestrates the training or evaluation of a DQN agent based on command-line arguments. 
    It handles model initialization, data loading, training/evaluation execution, result computation, and model saving.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing configuration parameters.
    """

    # Check for consistent action and cost definitions
    if len(args.actions) != len(args.cost):
        raise ValueError("The number of defined actions must match the number of associated costs.")

    # Initialize the DQN agent with specified parameters
    state_dim = 23  # State dimension (see RanSMAP paper for details)
    num_actions = len(args.actions)
    agent = Agent(state_dim=state_dim, num_actions=num_actions, lr=args.lr, gamma=args.gamma,
                   batch_size=args.batch_size, replay_size=args.replay_size,
                   init_epsilon=args.init_epsilon, final_epsilon=args.final_epsilon, exploration_steps=args.exploration_steps,
                   target_update_freq=args.target_update_freq, hidden_sizes=args.hidden_sizes,
                   log_dir=args.log_dir, duplicate_check=args.duplicate_check, distance_threshold=args.distance_threshold, 
                   beta=args.beta, rho=args.rho, eps=args.eps, sampling_type=args.sampling_type)

    # Create necessary directories for checkpoints, sample paths, trained models, and results
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_paths_dir, exist_ok=True)
    os.makedirs(args.trained_model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.mode == "Training":
        # Prepare training data paths, generating them if necessary
        training_yaml_path = os.path.join(args.sample_paths_dir, args.training_paths_name)
        if not os.path.exists(training_yaml_path):
            training_scenario = Scenario(args.scenario, args.train_path, args.mode, args.split_ratio)
            training_scenario.generate_paths()
            training_scenario.save_to_yaml(training_yaml_path)

        # Load training sample paths from the YAML file
        with open(training_yaml_path, 'r') as f:
            train_paths = yaml.safe_load(f)
            random.shuffle(train_paths)  # Shuffle training data for better generalization

        # Train the agent
        num_train_episodes = len(train_paths)
        agent.train(train_paths, num_train_episodes, args.base_reward, args.cost, max_engagement_length=args.max_engagement_length, 
                    T_max=args.T_max, T_window=args.T_window, checkpoint_freq=args.checkpoint_freq, checkpoint_dir=args.checkpoint_dir)
        print(f"Training complete!")

        # Save the trained model
        trained_model_path = os.path.join(args.trained_model_dir, args.model_name)
        agent.save(trained_model_path)
        print("Model saved.")

        # Remove checkpoints to free up disk space
        agent.delete_checkpoints(args.checkpoint_dir)

        # Compute and save evaluation results
        feature_importance = agent.dqn.get_feature_importance()
        compute_results(agent.y_true, agent.y_pred, agent.episode_rewards, agent.goal_completion, 
                        agent.loss, agent.mean_v, feature_importance, args.mode, args.results_dir, save=True)
        print(f"Results computed and saved in {args.results_dir}")

    elif args.mode == "Evaluation":
        # Load the pre-trained model for evaluation
        trained_model_path = os.path.join(args.trained_model_dir, args.model_name)
        if not os.path.exists(trained_model_path):
            raise ValueError("No trained model found. Please train a model before evaluation.")

        # Prepare evaluation data paths, generating them if necessary
        evaluation_yaml_path = os.path.join(args.sample_paths_dir, args.evaluation_paths_name)
        if not os.path.exists(evaluation_yaml_path):
            evaluation_scenario = Scenario(args.scenario, args.eval_path, args.mode, args.split_ratio)
            evaluation_scenario.generate_paths()
            evaluation_scenario.save_to_yaml(evaluation_yaml_path)

        # Load evaluation sample paths from the YAML file
        with open(evaluation_yaml_path, 'r') as f:
            eval_paths = yaml.safe_load(f)

        # Load the trained model weights
        agent.load(trained_model_path)
        print("Model loaded.")

        # Evaluate the agent's performance
        num_eval_episodes = len(eval_paths)
        eval_epsilon = args.eval_epsilon
        agent.evaluate(eval_paths, num_eval_episodes, args.base_reward, args.cost, max_engagement_length=args.max_engagement_length, 
                       eval_epsilon=eval_epsilon, T_max=args.T_max, T_window=args.T_window)
        print(f"Evaluation complete!")

        # Compute and save evaluation results
        compute_results(agent.y_true, agent.y_pred, agent.episode_rewards, agent.goal_completion, 
                        agent.loss, agent.mean_v, [], args.mode, args.results_dir, save=True)
        print(f"Results computed and saved in {args.results_dir}")

    else:
        raise ValueError("Invalid operation --mode specified. Choose either 'Training' or 'Evaluation'.")

if __name__ == "__main__":
    # Some of these parameters like batch_size, target_update_freq, replay_size,
    # exploration_steps, and lr may need adjustment with a larger number of samples
    parser = argparse.ArgumentParser()

    # Mode of operation
    parser.add_argument("--mode", type=str, choices=["Training", "Evaluation"], help="Mode of operation: Training or Evaluation")
    parser.add_argument("--scenario", type=str, choices=["split", "whole"], default="whole", help="Scenario of operation: split or whole (default: whole)")
    parser.add_argument("--split_ratio", type=float, default=0.75, help="Split ratio for training/evaluation dataset if scenario is split (default=0.75)")

    # Data and paths
    parser.add_argument("--train_path", type=str, default="/home/preet_derasari/Journal/datasets/RanSMAP/dataset/original", help="Path to training samples (default: /home/preet_derasari/Journal/datasets/RanSMAP/dataset/original)")
    parser.add_argument("--eval_path", type=str, default="/home/preet_derasari/Journal/datasets/RanSMAP/dataset/extra", help="Path to evaluation samples (default: /home/preet_derasari/Journal/datasets/RanSMAP/dataset/extra)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the results (default: results)")
    parser.add_argument("--sample_paths_dir", type=str, default="sample_paths", help="Directory to save sample paths YAML files (default: sample_paths)")
    parser.add_argument("--trained_model_dir", type=str, default="trained_models", help="Directory to save/load the trained model (default: trained_models)")
    parser.add_argument("--model_name", type=str, default="trained_model.pth", help="Name of the trained model file (default: trained_model.pth)")
    parser.add_argument("--training_paths_name", type=str, default="training_paths.yaml", help="Name of the YAML file to store training paths (default: training_paths.yaml)")
    parser.add_argument("--evaluation_paths_name", type=str, default="evaluation_paths.yaml", help="Name of the YAML file to store evaluation paths (default: evaluation_paths.yaml)")

    # Checkpointing and logging during training
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="Frequency of saving checkpoints in number of episodes (default: 100)")
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory to save tensorboard logs (default='runs')")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpointed_models", help="Directory to save checkpoints (default: checkpointed_models)")

    # DQN parameters
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[128, 128, 128], help="Number of layers and number of neurons per layer (default=[128, 64])")
    parser.add_argument("--lr", type=float, default=1, help="Learning rate (default=1)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor, reward + gamma should be [0,1] to keep stability (default=0.99)")
    parser.add_argument("--target_update_freq", type=int, default=2000, help="Update frequency of the target DQN (default=2000)")
    parser.add_argument("--eval_epsilon", type=float, default=0.05, help="Evaluation epsilon value (default=0.05)")
    parser.add_argument("--beta", type=float, default=0.01, help="Decay rate for reward (default=0.01)")
    parser.add_argument("--rho", type=float, default=0.95, help="Value of rho for Adadelta optimizer (default=0.01)")
    parser.add_argument("--eps", type=float, default=1e-6, help="Value of eps for Adadelta optimizer (default=1e-6)")
    parser.add_argument("--max_engagement_length", type=int, default=30, help="Consecutive steps the agent has to engage appropriately (default=30)")

    # Replay buffer parameters
    parser.add_argument("--duplicate_check", type=str, choices=["hash", "distance"], default="distance", help="How do you want to check for duplicate states, hash based or distance based (default: distance)")
    parser.add_argument("--distance_threshold", type=int, default=0.01, help="The euclidean distance value to use for checking duplicate states (default: 0.01)")
    parser.add_argument("--replay_size", type=int, default=50000, help="Replay buffer size (default=50000)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default=64)")
    parser.add_argument("--sampling_type", type=str, choices=["uniform", "priority"], default="uniform", help="How the experiences will be sampled, uniform distribution or recent priority (default: uniform)")

    # Eploration-exploitation parameters
    parser.add_argument("--init_epsilon", type=float, default=1.0, help="Initial epsilon value (default=1.0)")
    parser.add_argument("--final_epsilon", type=float, default=0.05, help="Final epsilon value (default=0.05)")
    parser.add_argument("--exploration_steps", type=int, default=60000, help="Number of steps to allow exploration (default=60000)")

    # Reward and cost parameters
    parser.add_argument("--actions", type=str, nargs="*", default=["continue", "defend"], help="List of actions (default=['continue', 'defend'])")
    parser.add_argument("--base_reward", type=float, default=0.1, help="Base reward, reward + gamma should be [0,1] to keep stability (default=0.1)")
    parser.add_argument("--cost", type=int, nargs="*", default=[1, 2], help="Cost list of each action (default=[1, 2])")

    # Parameters for adjusting the number of steps per episode (based on per-trace number of time windows = T_max / T_window)
    parser.add_argument("--T_max", type=int, default=30e9, help="Number seconds from the first timestamp (default=30e9 i.e., 30 seconds)")
    parser.add_argument("--T_window", type=int, default=1e8, help="Size of the time window (default=1e8 i.e., 0.1 seconds)")
    
    # ToDo: Figure out why and how to use it
    # parser.add_argument("--seed", type=int, default=0, help="Seed value(default=0)")

    args = parser.parse_args()
    
    main(args)  # Call the main function with the parsed arguments