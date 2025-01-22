# Automating proactive defense strategies using Deep Reinforcement Learning

This repository contains the code for the research paper "Automating proactive defense strategies using Deep Reinforcement Learning" (to be published).

This project uses Deep Reinforcement Learning to develop an agent that can automatically trigger proactive defense strategies against ransomware attacks.

Currently, the agent works in a "simulated" environment, where it observes memory and storage activity (HPC metric) from traces of ransomware and benign samples and learns how to deploy proactive defense strategies only when it observes malicious activity.

## Key Features and Novelties

* **Proactive Defense:**  Shifts the focus from reaction to prevention, minimizing the impact of ransomware attacks.
* **Adaptive Learning:** The DRL agent continuously adapts to evolving ransomware tactics, surpassing the limitations of signature-based methods.
* **Rich State Representation:**  Utilizes a 23-feature state space capturing detailed memory and storage access patterns for enhanced detection accuracy.
* **Reward-Driven Optimization:**  A tailored reward function guides the agent's learning, balancing effective defense with minimal false positives.
* **Flexible and Extensible:**  The provided codebase offers a modular and customizable framework for experimentation and further development.

## Codebase Overview

The project's codebase is structured as follows:

* **`main.py`:**  The main script for training and evaluating the DRL agent.  Handles command-line arguments, data loading, and experiment execution.
* **`agent.py`:** Implements the DQN agent, including the neural network architecture, action selection, reward computation, and optimization logic.
* **`scenario.py`:** Defines the training/evaluation scenarios and manages the interaction between the agent and the environment.
* **`state_computation.py`:**  Contains functions to calculate the state representation from raw system traces.
* **`results.py`:** Provides functions to compute and visualize evaluation metrics.

## Installation and Usage

1. **Clone the repository:**  `git clone <repository_url>`
2. **Install dependencies:**  Ensure you have the required Python packages installed (e.g., PyTorch, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, YAML. (More information in [Requirements](#requirements)).
3. **Prepare datasets:** Organize your training and evaluation datasets according to the expected format (More information in section [Dataset](#dataset)).
4. **Run training:**  Use `main.py` with appropriate arguments to train the DRL agent. Example: `python main.py --mode Training` (More information in [Training mode](#training-mode)).
5. **Run evaluation:**  After training, evaluate the agent's performance using: `python main.py --mode Evaluation` (More information in [Evaluation mode](#evaluation-mode)).
6. **Analyze results:**  Examine the generated metrics and visualizations in the `results` directory (More information in [Computing Metrics](#computing-metrics)).

## Requirements

The required Python packages can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Dataset

The RansMAP dataset used in this project can be found [here](https://github.com/manabu-hirano/RanSMAP).

It contains memory and storage access patterns for different ransomware and benign applications.

We use the samples in the `original` folder for training and the samples in the `extra` folder for evaluation.

The `load_data()` method in `agent.py` loads the csv files from each sample and returns the values as a pandas dataframe for computing the state of the environment after every `T_window` seconds. The 23 state features that are computed in `state_computation.py` are derived from the original RansMAP [paper](https://www.sciencedirect.com/science/article/pii/S0167404824005078?ref=pdf_download&fr=RR-2&rr=900f8853d9ad058b).

## Usage

To run the code, use the `main.py` script with the following arguments:

### Training mode

```bash
python main.py --mode Training --train_path /path/to/RanSMAP/dataset/original \\
               --T_d 50000000000 --T_window 100000000 --max_engagement_length 50 --base_reward 0.1 --gamma 0.8
```

*   `--mode Training`: Sets the mode to training.
*   `--train_path`: Path to the training samples.
*   `--T_d`: Total amount of trace time (in ns) the model will be trained using.
*   `--T_window`: Time period of each step that the agent takes (in ns)
*   `--max_engagement_length`: How many steps for the agent to end the episode (i.e., goal state)
*   `--base_reward`: Refer to the code in agent.py for more explanation
*   `--gamma`: Keep it so that the (maximum reward + gamma) should not exceed 1 for more stability.

### Evaluation mode

```bash
python main.py --mode Evaluation --eval_path /path/to/RanSMAP/dataset/extra \\
               --T_d 50000000000 --T_window 100000000 --max_engagement_length 50 --base_reward 0.1
```

*   `--mode Evaluation`: Sets the mode to evaluation.
*   `--eval_path`: Path to the evaluation samples.
*   `--T_d`: Total amount of trace time (in ns) the model will be evaluated on.
*   `--T_window`: Time period of each step that the agent takes (in ns). Explore this in future to see how it changes the results.
*   `--max_engagement_length`: How many steps for the agent to end the episode (i.e., goal state). Vary this in evaluation to observe how well the model was trained (e.g., 10%, 5%, 1% of `T_d`).
*   `--base_reward`: Use the same value with which you trained the model.

## Computing Metrics

The results.py script computes and visualizes key performance metrics for the DRL agent. It calculates standard classification metrics (accuracy, precision, recall, F1-score, specificity, AUC-ROC) using sklearn.metrics and generates a confusion matrix. Goal completion percentages, representing sustained proactive defense, are computed for all episodes, malicious episodes, and benign episodes, along with their distributions. Furthermore, it plots precision-recall curves, ROC curves, and episode rewards for comprehensive performance analysis. Feature importance, derived from the DQN's first layer weights, is also visualized. All results are saved as text files and plots within a specified directory.

## Technical Details

Here we describe some of the details about the DRL agent, including the DQN architecture, experience replay, state space, reward function, actions, optimizer and loss functions, checkpointing, and visualization.

### DRL Agent Architecture

The DRL agent in `agent.py` is designed for proactive ransomware detection and defense. It uses a deep Q-network (DQN) to learn a policy that maps system states to actions. Here's a breakdown of its architecture:

#### 1. Network Architecture (`DQN` class):

The DQN uses a fully connected neural network with the following structure:

* **Input Layer:** Takes a state vector of size 23 (representing system features) as input.
* **Hidden Layers:**  Three hidden layers, each with 128 neurons and Batch Normalization followed by ReLU activation. Dropout is applied after each Batch Normalization layer to prevent overfitting. The number of layers and neurons per layer can be customized using the `hidden_sizes` argument during agent initialization.
* **Output Layer:**  A linear layer with a softmax activation function, producing a probability distribution over the two possible actions: "continue" (0) and "defend" (1).

This architecture allows the network to learn complex non-linear relationships between system states and optimal actions.

#### 2. Experience Replay (`ReplayMemory` class):

The agent utilizes experience replay to improve learning stability and efficiency. Experiences are stored as tuples of (state, action, next_state, reward, done) in a cyclic buffer. Duplicate state checking (using either hashing or distance-based comparison) is implemented to avoid redundant data in the replay memory. During training, random batches of experiences are sampled from the replay memory to update the DQN.

#### 3. State Space

The state space is composed of 23 features extracted from memory and storage access patterns. These features include:

##### 3.1 Storage Features:

* Read and write throughput.
* Variance of read and write LBAs.
* Average Shannon entropy of write operations.

##### 3.2 Memory Features:

* Average Shannon entropy of write and read/write operations.
* Counts of 4KB and 2MB memory read, write, read/write, and execute operations.
* Counts of MMIO read, write, read/write, and execute operations.
* Variance of read, write, read/write, and execute GPAs.

#### 4. Target Network:

A target network, identical in architecture to the main DQN, is used to stabilize learning. The target network's weights are periodically updated by copying the weights from the main DQN. This helps to avoid oscillations and divergence during training.

#### 5. Action Selection:

The agent uses an epsilon-greedy strategy to balance exploration and exploitation.  With probability `epsilon`, a random action is chosen; otherwise, the action with the highest Q-value (predicted by the DQN) is selected. The `epsilon` value starts high and gradually decays over time, encouraging exploration early in training and exploitation later on.

#### 6. Reward Function:

The reward function is crucial for guiding the agent's learning. It rewards correct actions (defending against malicious samples, continuing with benign samples) and penalizes incorrect actions. The reward is also adjusted based on the current engagement length (number of consecutive positive rewards). This encourages the agent to maintain a proactive defense stance for a sustained period.

#### 7. Optimizer and Loss Function:

The agent uses the Adadelta optimizer to update the DQN's weights, and CrossEntropyLoss to compute the loss between predicted and target Q-values.

#### 8. Checkpointing:

The agent's training progress can be saved and resumed using checkpoints, which store the model weights, optimizer state, and other relevant information.  This allows for interruption and resumption of training without starting from scratch.

#### 9. TensorBoard Visualization

TensorBoard logs are saved in the 'runs' directory. To visualize training progress, run the command tensorboard --logdir runs in your terminal and open the provided URL in your web browser.

## Training Time

Training for 1440 samples takes approximately 18-20 hours without a GPU.

## Citation

Please cite our research paper as follows:

```
[Citation information will be added once the paper is published]
```