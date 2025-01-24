from results import compute_results
import pickle
import argparse
import os

def recompute(mode, results_dir):
    """
    Re-visualizes the results from saved data by calling compute_results.

    This function takes the saved data and calls the compute_results function
    to re-generate the visualizations.

    Args:
        data (dict): Dictionary containing the saved data.
        mode (str): Mode of operation ('Training' or 'Evaluation').
        results_dir (str): Directory to save the re-visualized results.
    """
    # Load the data
    if not os.path.exists(f"{results_dir}/{mode}_data.pkl"):
        raise FileNotFoundError(f"No data file found for mode '{mode}' in directory '{results_dir}'")

    with open(f"{results_dir}/{mode}_data.pkl", "rb") as file:
        try:
            data = pickle.load(file)
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Error loading data from file: {e}")
        
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    episode_rewards = data["episode_rewards"]
    goal_completion = data["goal_completion"]

    if mode == "Training":
        loss = data["loss"]
        mean_v = data["mean_v"]
        feature_importance = data["feature_importance"]
    elif mode == "Evaluation":
        loss = []
        mean_v = []
        feature_importance = []
    else:
        raise ValueError("Invalid mode. Training or Evaluation only!")

    # Call compute_results to re-generate visualizations
    compute_results(y_true, y_pred, episode_rewards, goal_completion, 
                    loss, mean_v, feature_importance, mode, results_dir, save=False)
    
    # ToDo: Alternatively we can add code here to visualize the saved data differently

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, help="Directory to load the results from (Required)")
    parser.add_argument("--mode", type=str, choices=["Training", "Evaluation"], help="Which mode of results need loading: Training or Evaluation")
    args = parser.parse_args()

    if any(arg is None for arg in (args.results_dir, args.mode)):
        raise ValueError("Need to specify both --mode and --results_dir!")
    
    if not os.path.exists(args.results_dir):
        raise ValueError("--results_dir does not exist")

    recompute(args.mode, args.results_dir)

    print(f"Re-computation of results from {args.results_dir} complete!")