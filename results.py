import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve # type: ignore
import pickle
import math

def save_data(data, filename):
    """
    Saves the provided data to a pickle file.

    Args:
        data (Any): The data to be saved. Can be any Python object.
        filename (str): The name of the file to save the data to.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Smoothen the data using exponential moving average. Alpha 0.999 means maximum smoothing. Derived from TensorBoard
def smooth_data(data, alpha=0.999):
    smoothed_data = []
    # last = data[0][1]  # Initialize EMA with the first data point
    last = 0
    num_acc = 0
    for ep, reward in data:
        smoothed_val = alpha * last + (1 - alpha) * reward
        last = smoothed_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if alpha != 1:
            debias_weight = 1 - math.pow(alpha, num_acc)
        smoothed_val = last / debias_weight
        smoothed_data.append((ep, smoothed_val))
    return smoothed_data

def compute_results(y_true, y_pred, episode_rewards, goal_completion, loss, mean_v, feature_importance, mode, results_dir, save=True):
    """
    Computes and saves various evaluation metrics and generates visualizations.

    This function calculates metrics like accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix,
    precision-recall curve, ROC curve, and goal completion percentage. It also generates and saves visualizations
    of these metrics, including histogram of completion points, action distribution, confusion matrix, precision-recall curve, ROC curve,
    episode rewards, training loss, training mean v, and feature importance.

    Args:
        y_true (list): List of true labels (0 for benign, 1 for malicious).
        y_pred (list): List of predicted actions (0 for continue, 1 for defend).
        episode_rewards (list): List of tuples, where each tuple contains (episode_number, reward, is_malicious).
        goal_completion (list): List of tuples, where each tuple contains (goal_completed, completion_point, is_malicious).
        loss (list): List of tuples, where each tuple contains (loss, steps_done).
        mean_v (list): List of tuples, where each tuple contains (mean_v, steps_done).
        feature_importance (np.ndarray): Feature importance scores.
        mode (str): Mode of operation ('Training' or 'Evaluation').
        results_dir (str): Directory to save the results.
        save (bool): Boolean value to save the data on disk.
    """
    # Calculate metrics after all episodes
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    auc_roc = roc_auc_score(y_true, y_pred)

    # Calculate goal completion percentages
    num_episodes = len(goal_completion)
    num_goal_completed = sum(goal_completed for goal_completed, _, _ in goal_completion)
    total_percentage = (num_goal_completed / num_episodes) * 100 if num_episodes != 0 else 0

    malicious_episodes = sum(is_malicious for _, _, is_malicious in goal_completion)
    malicious_goal_completed = sum(goal_completed and is_malicious for goal_completed, _, is_malicious in goal_completion)
    non_malicious_episodes = sum(not is_malicious for _, _, is_malicious in goal_completion)
    non_malicious_goal_completed = sum(goal_completed and not is_malicious for goal_completed, _, is_malicious in goal_completion)

    malicious_percentage = (malicious_goal_completed / malicious_episodes) * 100 if malicious_episodes > 0 else 0
    non_malicious_percentage = (non_malicious_goal_completed / non_malicious_episodes) * 100 if non_malicious_episodes > 0 else 0

    # Save metrics to file
    metrics_file_path = f"{results_dir}/{mode}_metrics.txt"
    with open(metrics_file_path, "w") as f:
        f.write(f"{mode} Metrics:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  Specificity: {specificity:.4f}\n")
        f.write(f"  F1-score: {f1:.4f}\n")
        f.write(f"  AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"Goal Completion Percentage:\n")
        f.write(f"  Total Percentage: {total_percentage:.2f}%\n")
        f.write(f"  Malicious Episodes: {malicious_percentage:.2f}%\n")
        f.write(f"  Non-Malicious Episodes: {non_malicious_percentage:.2f}%\n")

    # Generate and save visualizations

    # Get the histogram of percentage of episodes where goal is reached at what point in the episode
    total_completion_points = []
    malicious_completion_points = []
    benign_completion_points = []

    for goal_completed, completion_point, is_malicious in goal_completion:
        if goal_completed:
            total_completion_points.append(completion_point * 100)
            if is_malicious:
                malicious_completion_points.append(completion_point * 100)
            else:
                benign_completion_points.append(completion_point * 100)
    
    bin_edges = np.arange(0, 110, 10)
    bin_labels = [f"{i}-{i+10}%" for i in range(0, 100, 10)]

    if total_completion_points or malicious_completion_points or benign_completion_points:
        completion_histogram_path = f"{results_dir}/{mode}_goal_completion_combined_histogram.png"

        # Create a pandas DataFrame to organize the data
        data = pd.DataFrame({
            'Completion Percentage': total_completion_points + malicious_completion_points + benign_completion_points,
            'Episode Type': ['All Episodes'] * len(total_completion_points) + \
                            ['Malicious Episodes'] * len(malicious_completion_points) + \
                            ['Benign Episodes'] * len(benign_completion_points)
        })

        plt.figure(figsize=(8, 6))

        # Use seaborn's histplot with multiple='dodge' for grouped bars
        ax = sns.histplot(
            data,
            x='Completion Percentage',
            hue='Episode Type',
            bins=bin_edges,
            stat="probability",
            discrete=False,
            shrink=0.8,
            multiple='dodge',  # This separates the bars
            palette={'All Episodes': 'blue', 'Malicious Episodes': 'red', 'Benign Episodes': 'green'}
        )

        plt.xticks(bin_edges[:-1], bin_labels, rotation=45, ha="left")
        plt.xlabel("Distribution Goal Completion Points")
        plt.ylabel("Percentage of Episodes Where the Goal was Reached")
        plt.title("Goal Completion Point in an Episode")

        ax.yaxis.set_major_formatter(PercentFormatter(1))
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(completion_histogram_path)
        plt.close()

    # Action Distribution
    action_distribution_path = f"{results_dir}/{mode}_action_distribution.png"
    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred, bins=2, discrete=True, stat="probability", shrink=0.8)
    plt.xticks([0, 1], ["Continue", "Defend"])
    plt.xlabel("Action")
    plt.ylabel("Probability")
    plt.title("Action Distribution")
    plt.savefig(action_distribution_path)
    plt.close()

    # Confusion Matrix
    confusion_matrix_path = f"{results_dir}/{mode}_confusion_matrix.png"
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Continue", "Defend"], yticklabels=["Benign", "Malware"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_path)
    plt.close()

    # Precision-Recall Curve
    pr_curve_path = f"{results_dir}/{mode}_precision_recall_curve.png"
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(pr_curve_path)
    plt.close()
    
    # ROC Curve
    roc_curve_path = f"{results_dir}/{mode}_roc_curve.png"
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig(roc_curve_path)
    plt.close()

    # Plot episode rewards
    episode_rewards_path = f"{results_dir}/{mode}_episode_rewards.png"
    plt.figure(figsize=(10, 6))
    # Separate rewards based on episode type
    malicious_rewards = [(ep, reward) for ep, reward, is_malicious in episode_rewards if is_malicious]
    benign_rewards = [(ep, reward) for ep, reward, is_malicious in episode_rewards if not is_malicious]
    # Smooth the rewards
    smoothed_all_rewards = smooth_data([(ep, reward) for ep, reward, _ in episode_rewards])
    smoothed_malicious_rewards = smooth_data(malicious_rewards)
    smoothed_benign_rewards = smooth_data(benign_rewards)
    # Plot the original data with transparency
    plt.plot(*zip(*[(ep, reward) for ep, reward, _ in episode_rewards]), label="All Episodes (Original)", color='blue', alpha=0.2)  # All data
    plt.plot(*zip(*malicious_rewards), label="Malicious Episodes (Original)", color='red', alpha=0.2)  # Malicious
    plt.plot(*zip(*benign_rewards), label="Benign Episodes (Original)", color='green', alpha=0.2)  # Benign
    # Plot the smoothed rewards
    plt.plot(*zip(*smoothed_all_rewards), label="All Episodes (Smoothed)", color='blue')
    plt.plot(*zip(*smoothed_malicious_rewards), label="Malicious Episodes (Smoothed)", color='red')
    plt.plot(*zip(*smoothed_benign_rewards), label="Benign Episodes (Smoothed)", color='green')
    plt.xlabel(f"{mode} Episodes")
    plt.ylabel("Reward")
    plt.title(f"Trend of Rewards During {mode}")
    plt.legend()
    plt.savefig(episode_rewards_path)
    plt.close()

    if mode == "Training":
        # Plot training loss
        loss_path = f"{results_dir}/{mode}_Loss.png"
        plt.figure(figsize=(10, 6))
        # Smooth the loss
        smoothed_loss = smooth_data(loss)
        # Plot the original data with transparency
        plt.plot(*zip(*loss), label="Loss (Original)", alpha=0.2)
        # Plot the smoothed loss
        plt.plot(*zip(*smoothed_loss), label="Loss (Smoothed)")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(loss_path)
        plt.close()

        # Plot Mean V values
        mean_v_path = f"{results_dir}/{mode}_mean_v.png"
        plt.figure(figsize=(10, 6))
        # Smooth the Mean V
        smoothed_mean_v = smooth_data(mean_v)
        # Plot the original data with transparency
        plt.plot(*zip(*mean_v), label="Mean V (Original)", alpha=0.2)
        # Plot the smoothed Mean V
        plt.plot(*zip(*smoothed_mean_v), label="Mean V (Smoothed)")
        plt.xlabel("Training Steps")
        plt.ylabel("Mean V")
        plt.title("Mean values of V(s)")
        plt.savefig(mean_v_path)
        plt.close()

        # Feature Importance
        feature_importance_path = f"{results_dir}/feature_importance.png"
        feature_names = ["T_sr", "T_sw", "V_sr", "V_sw", "H_sw", "H_mw", "H_mrw", 
                            "C_4KBr", "C_4KBw", "C_4KBrw", "C_4KBx", "C_2MBr", "C_2MBw", 
                            "C_2MBrw", "C_2MBx", "C_MMIOr", "C_MMIOw", "C_MMIOrw", "C_MMIOx", 
                            "V_mr", "V_mw", "V_mrw", "V_mx"]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xticks(range(len(feature_importance)), feature_names, rotation=45)  # Use feature names as labels
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.title("Importance Score of 23 features used to formulate the agent's state")
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.savefig(feature_importance_path)
        plt.close()

    if save:
        if mode == "Training":
            # Save relevant data for future re-visualization
            data_to_save = {
                "y_true": y_true,
                "y_pred": y_pred,
                "episode_rewards": episode_rewards,
                "goal_completion": goal_completion,
                "feature_importance": feature_importance,
                "loss":loss,
                "mean_v":mean_v
            }
            save_data(data_to_save, f"{results_dir}/{mode}_data.pkl")
        else:
            # Save relevant data for future re-visualization
            data_to_save = {
                "y_true": y_true,
                "y_pred": y_pred,
                "episode_rewards": episode_rewards,
                "goal_completion": goal_completion
            }
            save_data(data_to_save, f"{results_dir}/{mode}_data.pkl")