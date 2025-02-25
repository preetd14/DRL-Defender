from results import compute_results, smooth_data
import pickle
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

training_paths = [
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-75"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-25"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-5"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-50"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-75"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-25"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-5"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-50"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-75"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-25"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-5"),
    ("Training", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-50"),
]

evaluation_paths = [
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-75/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-75/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-75/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-75/Eval_len-75"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-25/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-25/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-25/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-5/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-5/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-5/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-50/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-50/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.99_base-0.006/Train_len-50/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-75/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-75/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-75/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-75/Eval_len-75"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-25/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-25/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-25/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-25/Eval_len-75"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-5/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-5/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-5/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-5/Eval_len-75"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-50/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-50/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-50/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.5_base-0.24/Train_len-50/Eval_len-75"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-75/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-75/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-75/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-75/Eval_len-75"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-25/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-25/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-25/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-5/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-5/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-5/Eval_len-25"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-50/Eval_len-50"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-50/Eval_len-5"),
    ("Evaluation", "results/Train-original_Eval-original/sampling-uniform/gamma-0.01_base-0.6/Train_len-50/Eval_len-25"),
]

def recompute_combined(store_dir):
    """
    Generates combined charts for episode rewards, goal completion points, mean_v, and loss from the paths stored in training_paths variable.

    Args:
        store_dir (str): Directory to save the combined visualizations.
    """
    def get_combined_data(paths, key_index, filter_func):
        combined_data = {
            "episode_rewards": {},
            "goal_completion": {},
            "loss": {},
            "mean_v": {}
        }
        for mode, path in paths:
            if mode != "Training":
                continue
            if not filter_func(path):
                continue
            data_file = f"{path}/{mode}_data.pkl"
            if not os.path.exists(data_file):
                continue
            with open(data_file, "rb") as file:
                try:
                    data = pickle.load(file)
                except (pickle.UnpicklingError, EOFError) as e:
                    print(f"Error loading data from file {data_file}: {e}")
                    continue
                key = path.split('/')[key_index]  # Use the last part of the path as the key
                if key not in combined_data["episode_rewards"]:
                    combined_data["episode_rewards"][key] = []
                    combined_data["goal_completion"][key] = []
                    combined_data["loss"][key] = []
                    combined_data["mean_v"][key] = []
                combined_data["episode_rewards"][key].extend(data["episode_rewards"])
                combined_data["goal_completion"][key].extend(data["goal_completion"])
                combined_data["loss"][key].extend(data["loss"])
                combined_data["mean_v"][key].extend(data["mean_v"])
        return combined_data

    def plot_combined_episode_rewards(combined_data, title, filename, labels, colors):
        # plt.figure(figsize=(2.33, 1.4))
        plt.figure(figsize=(5, 3))
        for key, rewards in combined_data["episode_rewards"].items():
            smoothed_rewards = smooth_data([(ep, reward) for ep, reward, _ in rewards])
            plt.plot(*zip(*[(ep, reward) for ep, reward, _ in rewards]), color=f"{colors[key]}", alpha=0.2)
            plt.plot(*zip(*smoothed_rewards), color=f"{colors[key]}", label=f"{labels[key]}")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        # plt.title(title)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, 
                   columnspacing=1.1, handlelength=1.0, handletextpad=0.2,
                   shadow=True)  # Move legend above plot and print horizontally
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_combined_goal_completion(combined_data, title, filename, labels, colors):
        # plt.figure(figsize=(2.33, 1.4))
        plt.figure(figsize=(5, 3))
        all_data = []
        for key, completions in combined_data["goal_completion"].items():
            total_completion_points = [completion_point * 100 for goal_completed, completion_point, _ in completions if goal_completed]
            all_data.extend([(completion_point, key) for completion_point in total_completion_points])
        
        if all_data:
            data = pd.DataFrame(all_data, columns=['Completion Percentage', 'Episode Type'])
            bin_edges = np.arange(0, 110, 10)
            bin_labels = [f"{i}-{i+10}%" for i in range(0, 100, 10)]
            sns.histplot(
                data,
                x='Completion Percentage',
                hue='Episode Type',
                bins=bin_edges,
                stat="probability",
                discrete=False,
                shrink=0.8,
                multiple='dodge',
                palette=colors  # Use the colors dictionary for the plot
            )
            plt.xticks(bin_edges[:-1], bin_labels, rotation=45, ha="left")
            plt.xlabel("Distribution Goal Completion Points")
            plt.ylabel("% of Episodes")
            # plt.title(title)
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            # Manually create the legend
            handles = [plt.Line2D([0], [0], color=colors[key], lw=4) for key in combined_data["goal_completion"].keys()]
            plt.legend(handles=handles, labels=[labels[key] for key in combined_data["goal_completion"].keys()],
                    loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, 
                    columnspacing=1.1, handlelength=1.0, handletextpad=0.2,
                    shadow=True)  # Move legend above plot and print horizontally
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

    def plot_combined_loss(combined_data, title, filename, labels, colors):
        # plt.figure(figsize=(2.33, 1.4))
        plt.figure(figsize=(5, 3))
        for key, losses in combined_data["loss"].items():
            smoothed_loss = smooth_data(losses)
            plt.plot(*zip(*losses), alpha=0.2, color=f"{colors[key]}")
            plt.plot(*zip(*smoothed_loss), color=f"{colors[key]}", label=f"{labels[key]}")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        # plt.title(title)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, 
                   columnspacing=1.1, handlelength=1.0, handletextpad=0.2,
                   shadow=True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_combined_mean_v(combined_data, title, filename, labels, colors):
        # plt.figure(figsize=(2.33, 1.4))
        plt.figure(figsize=(5, 3))
        for key, mean_vs in combined_data["mean_v"].items():
            smoothed_mean_v = smooth_data(mean_vs)
            plt.plot(*zip(*mean_vs), alpha=0.2, color=f"{colors[key]}")
            plt.plot(*zip(*smoothed_mean_v), color=f"{colors[key]}", label=f"{labels[key]}")
        plt.xlabel("Training Steps")
        plt.ylabel("Mean V")
        # plt.title(title)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, 
                   columnspacing=1.1, handlelength=1.0, handletextpad=0.2,
                   shadow=True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # Combination of gamma and base reward values
    gamma_base_combinations = [
        ("Train_len-75", ["gamma-0.99_base-0.006", "gamma-0.5_base-0.24", "gamma-0.01_base-0.6"]),
        ("Train_len-50", ["gamma-0.99_base-0.006", "gamma-0.5_base-0.24", "gamma-0.01_base-0.6"]),
        ("Train_len-25", ["gamma-0.99_base-0.006", "gamma-0.5_base-0.24", "gamma-0.01_base-0.6"])
    ]
    gamma_base_labels = {
        "gamma-0.99_base-0.006": r"$\it{\gamma}:0.99,\it{r}:0.006$",
        "gamma-0.5_base-0.24": r"$\it{\gamma}:0.5,\it{r}:0.24$",
        "gamma-0.01_base-0.6": r"$\it{\gamma}:0.01,\it{r}:0.6$"
    }
    gamma_base_colors = {
        "gamma-0.99_base-0.006": "tab:blue",
        "gamma-0.5_base-0.24": "tab:orange",
        "gamma-0.01_base-0.6": "tab:red"
    }
    key_index = -2
    for length, gamma_bases in gamma_base_combinations:
        combined_data = get_combined_data(training_paths, key_index, lambda path: any(gb in path for gb in gamma_bases) and length in path)
        plot_combined_episode_rewards(combined_data, f"Combined Episode Rewards for {length}", f"{store_dir}/combined_episode_rewards_{length}.png", gamma_base_labels, gamma_base_colors)
        plot_combined_goal_completion(combined_data, f"Combined Goal Completion Points for {length}", f"{store_dir}/combined_goal_completion_{length}.png", gamma_base_labels, gamma_base_colors)
        plot_combined_loss(combined_data, f"Combined Training Loss for {length}", f"{store_dir}/combined_loss_{length}.png", gamma_base_labels, gamma_base_colors)
        plot_combined_mean_v(combined_data, f"Combined Mean V for {length}", f"{store_dir}/combined_mean_v_{length}.png", gamma_base_labels, gamma_base_colors)

    # Combination of Training lengths
    gamma_base_lengths = [
        ("gamma-0.99_base-0.006", ["Train_len-75", "Train_len-50", "Train_len-25"]),
        ("gamma-0.5_base-0.24", ["Train_len-75", "Train_len-50", "Train_len-25"]),
        ("gamma-0.01_base-0.6", ["Train_len-75", "Train_len-50", "Train_len-25"])
    ]
    length_labels = {
        "Train_len-75": r"$\it{L}_{\mathrm{max}}:15\%$",
        "Train_len-50": r"$\it{L}_{\mathrm{max}}:10\%$",
        "Train_len-25": r"$\it{L}_{\mathrm{max}}:5\%$"
    }
    length_colors = {
        "Train_len-75": "tab:blue",
        "Train_len-50": "tab:orange",
        "Train_len-25": "tab:red"
    }
    key_index = -1
    for gamma_base, lengths in gamma_base_lengths:
        combined_data = get_combined_data(training_paths, key_index, lambda path: gamma_base in path and any(length in path for length in lengths))
        plot_combined_episode_rewards(combined_data, f"Combined Episode Rewards for {gamma_base}", f"{store_dir}/combined_episode_rewards_{gamma_base}.png", length_labels, length_colors)
        plot_combined_goal_completion(combined_data, f"Combined Goal Completion Points for {gamma_base}", f"{store_dir}/combined_goal_completion_{gamma_base}.png", length_labels, length_colors)
        plot_combined_loss(combined_data, f"Combined Training Loss for {gamma_base}", f"{store_dir}/combined_loss_{gamma_base}.png", length_labels, length_colors)
        plot_combined_mean_v(combined_data, f"Combined Mean V for {gamma_base}", f"{store_dir}/combined_mean_v_{gamma_base}.png", length_labels, length_colors)

def recompute(results_type, results_dir):
    """
    Re-visualizes the results from saved data by calling compute_results.

    This function takes the saved data and calls the compute_results function
    to re-generate the visualizations.

    Args:
        data (dict): Dictionary containing the saved data.
        mode (str): Mode of operation ('individual' or 'combined').
        results_type (str): Type of results to load ('Training' or 'Evaluation').
        results_dir (str): Directory to save the re-visualized results.
    """
    # Load the data
    if not os.path.exists(f"{results_dir}/{results_type}_data.pkl"):
        raise FileNotFoundError(f"No data file found for results_type '{results_type}' in directory '{results_dir}'")

    with open(f"{results_dir}/{results_type}_data.pkl", "rb") as file:
        try:
            data = pickle.load(file)
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Error loading data from file: {e}")
        
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    episode_rewards = data["episode_rewards"]
    goal_completion = data["goal_completion"]

    if results_type == "Training":
        loss = data["loss"]
        mean_v = data["mean_v"]
        feature_importance = data["feature_importance"]
    elif results_type == "Evaluation":
        loss = []
        mean_v = []
        feature_importance = []
    else:
        raise ValueError("Invalid results_type. Training or Evaluation only!")

    # Call compute_results to re-generate visualizations
    compute_results(y_true, y_pred, episode_rewards, goal_completion, 
                    loss, mean_v, feature_importance, results_type, results_dir, save=False)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["individual", "combined"], help="Mode of operation: individual results or combined (Required)")
    parser.add_argument("--results_dir", type=str, help="Directory to load the results from (Required if mode is individual)")
    parser.add_argument("--results_type", type=str, choices=["Training", "Evaluation"], help="Which type of results need loading: Training or Evaluation (Required if mode is individual)")
    parser.add_argument("--store_dir", type=str, help="Directory to store the results to (Required if mode is combined)")
    args = parser.parse_args()

    if args.mode is None:
        raise ValueError("Need to specify --mode!")

    if args.mode == "individual" and args.results_type is None:
        raise ValueError("Need to specify --results_type when using mode 'individual'!")
    
    if args.mode == "individual" and not os.path.exists(args.results_dir):
        raise ValueError("--results_dir does not exist")
    
    if args.mode == "combined" and not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)

    if args.mode == "individual":
        recompute(args.results_type, args.results_dir)
    elif args.mode == "combined":
        recompute_combined(args.store_dir)
    else:
        raise ValueError("Invalid mode. Choose 'individual' or 'combined'")
    
    print("Re-computation complete!")