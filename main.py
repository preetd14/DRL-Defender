from agent import Agent
import argparse
import yaml
import os
import random
from scenario import Scenario

def main(args):
    """
    Main function to run the DQN agent in training or evaluation mode.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # Check if the number of actions and costs match
    if len(args.actions) != len(args.cost):
        raise ValueError("Number of actions and costs must be equal")

    # Initialize agent
    state_dim = 23  # ToDo: parameterize this if possible
    num_actions = len(args.actions)
    agent = Agent(state_dim=state_dim, num_actions=num_actions, lr=args.lr, gamma=args.gamma,
                   batch_size=args.batch_size, replay_size=args.replay_size,
                   init_epsilon=args.init_epsilon, final_epsilon=args.final_epsilon, exploration_steps=args.exploration_steps,
                   target_update_freq=args.target_update_freq, hidden_sizes=args.hidden_sizes,
                   log_dir=args.log_dir, duplicate_check=args.duplicate_check, distance_threshold=args.distance_threshold)

    # Create directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_paths_dir, exist_ok=True)
    os.makedirs(args.trained_model_dir, exist_ok=True)

    if args.mode == "train":
        # Check if YAML files exist, generate if not
        training_yaml_path = os.path.join(args.sample_paths_dir, "training_paths.yaml")
        if not os.path.exists(training_yaml_path):
            training_scenario = Scenario("train", args.train_path)
            training_scenario.generate_paths()
            training_scenario.save_to_yaml(training_yaml_path)

        # Load sample paths from YAML files
        with open(training_yaml_path, 'r') as f:
            train_paths = yaml.safe_load(f)
            random.shuffle(train_paths)

        num_train_episodes = len(train_paths)
        total_reward = agent.train(train_paths, num_train_episodes, args.base_reward, args.cost, T_d=args.T_d, T_window=args.T_window,
                                    checkpoint_freq=args.checkpoint_freq, checkpoint_dir=args.checkpoint_dir)
        print(f"Training complete! Total reward: {total_reward}")

        trained_model_path = os.path.join(args.trained_model_dir, "trained_model.pth")
        agent.save(trained_model_path)
        print("Model saved.")
        # Delete checkpoints after training completes
        agent.delete_checkpoints(args.checkpoint_dir)

    elif args.mode == "eval":
        # Check if a trained model exists
        trained_model_path = os.path.join(args.trained_model_dir, "trained_model.pth")
        if not os.path.exists(trained_model_path):
            raise ValueError("No trained model found. Please train the model first.")

        # Check if YAML files exist, generate if not
        evaluation_yaml_path = os.path.join(args.sample_paths_dir, "evaluation_paths.yaml")
        if not os.path.exists(evaluation_yaml_path):
            evaluation_scenario = Scenario("eval", args.eval_path)
            evaluation_scenario.generate_paths()
            evaluation_scenario.save_to_yaml(evaluation_yaml_path)

        # Load sample paths from YAML files
        with open(evaluation_yaml_path, 'r') as f:
            eval_paths = yaml.safe_load(f)

        agent.load(trained_model_path)
        print("Model loaded.")
        num_eval_episodes = len(eval_paths)
        eval_epsilon = args.eval_epsilon
        total_reward = agent.evaluate(eval_paths, num_eval_episodes, args.base_reward, args.cost, eval_epsilon=eval_epsilon, T_d=args.T_d, T_window=args.T_window)
        print(f"Evaluation complete! Total reward: {total_reward}")

    else:
        raise ValueError("Invalid mode. Use 'train' OR 'eval'.")

if __name__ == "__main__":
    # Some of these parameters like batch_size, target_update_freq, replay_size,
    # exploration_steps, and lr may need adjustment with a larger number of samples
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], help="Mode of operation: train or eval")
    parser.add_argument("--train_path", type=str, default="/home/preet_derasari/Journal/datasets/RanSMAP/dataset/original",
                        help="Path to training samples (default: /home/preet_derasari/Journal/datasets/RanSMAP/dataset/original)")
    parser.add_argument("--eval_path", type=str, default="/home/preet_derasari/Journal/datasets/RanSMAP/dataset/extra",
                        help="Path to evaluation samples (default: /home/preet_derasari/Journal/datasets/RanSMAP/dataset/extra)")
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[128, 64],
                        help="Number of layers and number of neurons per layer (default=[128, 64])")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default=0.001)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default=32)")
    parser.add_argument("--target_update_freq", type=int, default=2000, help="Update freq of the target DQN(default=2000)")
    # parser.add_argument("--seed", type=int, default=0, help="Seed value(default=0)") # ToDo: Figure out why and how to use it
    parser.add_argument("--replay_size", type=int, default=50000, help="Replay buffer size (default=50000)")
    parser.add_argument("--eval_epsilon", type=float, default=0.05, help="Evaluation epsilon value(default=0.05)")
    parser.add_argument("--final_epsilon", type=float, default=0.05, help="Final epsilon value(default=0.05)")
    parser.add_argument("--init_epsilon", type=float, default=1.0, help="Initial epsilon value(default=1.0)")
    parser.add_argument("--exploration_steps", type=int, default=60000,
                        help="Number of steps to allow exploration (default=60000)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor(default=0.95)")
    parser.add_argument("--T_d", type=int, default=30e9,
                        help="Number seconds from the first timestamp (default=30e9 i.e., 30 seconds)")
    parser.add_argument("--T_window", type=int, default=1e8, help="Size of the time window (default=1e8 i.e., 0.1 seconds)")
    parser.add_argument("--actions", type=str, nargs="*", default=["continue", "defend"],
                        help="List of actions (default=['continue', 'defend'])")
    parser.add_argument("--base_reward", type=int, default=3, help="Base reward (default=3)")
    parser.add_argument("--cost", type=int, nargs="*", default=[1, 2], help="Cost list of each action (default=[1, 2])")
    parser.add_argument("--checkpoint_freq", type=int, default=100,
                        help="Frequency of saving checkpoints in number of episodes (default: 100)")
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory to save tensorboard logs (default='runs')")
    parser.add_argument("--trained_model_dir", type=str, default="trained_models",
                        help="Directory to save/load the trained model (default: trained_models)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpointed_models",
                        help="Directory to save checkpoints (default: checkpointed_models)")
    parser.add_argument("--sample_paths_dir", type=str, default="sample_paths",
                        help="Directory to save sample paths YAML files (default: sample_paths)")
    parser.add_argument("--duplicate_check", type=str, choices=["hash", "distance"], default="distance", help="How do you want to check for duplicates, hash based or distance based (default: distance)")
    parser.add_argument("--distance_threshold", type=int, default=0.01, help="The euclidean distance value to use for checking duplicates (default: 0.01)")
    args = parser.parse_args()
    
    main(args)  # Call the main function with the parsed arguments