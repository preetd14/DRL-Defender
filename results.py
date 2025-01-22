import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve # type: ignore

def compute_results(y_true, y_pred, episode_rewards, malicious_rewards, benign_rewards, goal_completion, feature_importance, mode, results_dir):
    """
    Computes and saves various evaluation metrics and generates visualizations.

    This function calculates metrics like accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix,
    precision-recall curve, ROC curve, and goal completion percentage. It also generates and saves visualizations
    of these metrics, including action distribution, confusion matrix, precision-recall curve, ROC curve,
    episode rewards, and feature importance.

    Args:
        y_true (list): List of true labels (0 for benign, 1 for malicious).
        y_pred (list): List of predicted actions (0 for continue, 1 for defend).
        episode_rewards (list): List of tuples, where each tuple contains (episode_number, reward).
        malicious_rewards (list): List of tuples, where each tuple contains (episode_number, reward) for malicious samples.
        benign_rewards (list): List of tuples, where each tuple contains (episode_number, reward) for benign samples.
        goal_completion (list): List of tuples, where each tuple contains (goal_completed, is_malicious).
        feature_importance (np.ndarray): Feature importance scores.
        mode (str): Mode of operation ('Training' or 'Evaluation').
        results_dir (str): Directory to save the results.
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
    
    bin_edges = np.arange(0, 101, 10)
    bin_labels = [f"{bin_start}-{bin_end}%" for bin_start, bin_end in zip(bin_edges[:-1], bin_edges[1:])]

    if total_completion_points: # Check if there is any data to plot
        # Plot total histogram
        tot_histogram_path = f"{results_dir}/{mode}_goal_completion_total_histogram.png"
        plt.figure(figsize=(6, 4))
        sns.histplot(total_completion_points, bins=bin_edges, stat="probability", discrete=False)
        plt.xticks(bin_edges[:-1], bin_labels, rotation=45, ha="right")
        plt.xlabel("Goal Completion Percentage (All Episodes)")
        plt.ylabel("Number of Episodes")
        plt.title("Goal Completion Distribution (All Episodes)")
        plt.savefig(tot_histogram_path)
        plt.close()

    if malicious_completion_points: # Check if there is any data to plot
        # Plot malicious histogram
        mal_histogram_path = f"{results_dir}/{mode}_goal_completion_malicious_histogram.png"
        plt.figure(figsize=(6, 4))
        sns.histplot(malicious_completion_points, bins=bin_edges, stat="probability", discrete=False)
        plt.xticks(bin_edges[:-1], bin_labels, rotation=45, ha="right")
        plt.xlabel("Goal Completion Percentage (Malicious Episodes)")
        plt.ylabel("Number of Episodes")
        plt.title("Goal Completion Distribution (Malicious Episodes)")
        plt.savefig(mal_histogram_path)
        plt.close()

    if benign_completion_points: # Check if there is any data to plot
        # Plot benign histogram
        ben_histogram_path = f"{results_dir}/{mode}_goal_completion_benign_histogram.png"
        plt.figure(figsize=(6, 4))
        sns.histplot(benign_completion_points, bins=bin_edges, stat="probability", discrete=False)
        plt.xticks(bin_edges[:-1], bin_labels, rotation=45, ha="right")
        plt.xlabel("Goal Completion Percentage (Benign Episodes)")
        plt.ylabel("Number of Episodes")
        plt.title("Goal Completion Distribution (Benign Episodes)")
        plt.savefig(ben_histogram_path)
        plt.close()

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
    plt.plot(*zip(*episode_rewards), label="All Samples")
    plt.plot(*zip(*malicious_rewards), label="Malicious")
    plt.plot(*zip(*benign_rewards), label="Non-Malicious")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.legend()
    plt.savefig(episode_rewards_path)
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
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")
    plt.title("Feature Importance")
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.savefig(feature_importance_path)
    plt.close()
