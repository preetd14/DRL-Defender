# Automating proactive defense strategies using Deep Reinforcement Learning

This repository contains the code for the research paper "Automating proactive defense strategies using Deep Reinforcement Learning" (to be published).

This project uses Deep Reinforcement Learning to develop an agent that can automatically trigger proactive defense strategies against ransomware attacks.

Currently, the agent works in a "simulated" environment, where it observes memory and storage activity (HPC metric) from traces of ransomware and benign samples and learns how to deploy proactive defense strategies only when it observes malicious activity.

## Technical Details

Here we describe some of the details about the DRL agent, including the DQN architecture, state space, reward function, actions, hyperparameters, and visualization.

### DQN Architecture

We use the DQN algrotithm to develop the agent. The DQN consists of 3 fully connected layers with ReLU activation functions. The input layer has 23 neurons, followed by two hidden layers with 128 and 64 neurons respectively. The output layer has 2 neurons, one for each action (continue, defend). We can modify the number of layers and neurons per layer using the command line option `--hidden_sizes` in `main.py`.

### State Space

The state space is composed of 23 features extracted from memory and storage access patterns. These features include:

#### Storage Features:

* Read and write throughput.
* Variance of read and write LBAs.
* Average Shannon entropy of write operations.

#### Memory Features:

* Average Shannon entropy of write and read/write operations.
* Counts of 4KB and 2MB memory read, write, read/write, and execute operations.
* Counts of MMIO read, write, read/write, and execute operations.
* Variance of read, write, read/write, and execute GPAs.

### Actions

Currently we support simulation of two actions, "continue" and "defend".

* Continue: This action represents situations where the sample does not exhibit any malicious activity and the agent wants to resume the execution of the application (i.e., a benign application or during the benign phase of a malware).
* Defend: This action represents situations where the sample exhibits malicious activity and the agent wants to defend against that activity using proactive defense strategies.

### Reward Function

The reward function is designed to encourage the agent to take defensive actions when facing malicious traces while minimizing the cost of unnecessary defenses. The agent receives:

* A positive reward for correctly identifying and defending against malicious traces.
* A negative reward for incorrect actions, such as continuing when facing a malicious trace or defending against a benign trace.

### Hyperparameter Tuning

The DQN agent was trained using the Adam optimizer with a learning rate of 0.001 and a discount factor of 0.99. The batch size was set to 32 and an epsilon-greedy exploration strategy was used with an initial exploration rate of 1.0, decaying linearly to 0.05 over 60,000 steps.

### Evaluation Metrics

The trained agent's performance is evaluated on a separate set of unseen samples. The evaluation metrics used are accuracy, precision, recall, and F1-score, which measure the agent's ability to correctly classify and defend against malicious traces.

### TensorBoard Visualization

TensorBoard logs are saved in the 'runs' directory. To visualize training progress, run the command tensorboard --logdir runs in your terminal and open the provided URL in your web browser.

## Installation

Clone this repository and install the required packages (see Requirements).

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
python main.py --mode train --train_path /path/to/RanSMAP/dataset/original \\
               --actions continue defend --cost 1 2
```

*   `--mode train`: Sets the mode to training.
*   `--train_path`: Path to the training samples.
*   `--actions`: List of actions the agent can take.
*   `--cost`: Cost of each action.

### Evaluation mode

```bash
python main.py --mode eval --eval_path /path/to/RanSMAP/dataset/extra \\
               --actions continue defend --cost 1 2
```

*   `--mode eval`: Sets the mode to evaluation.
*   `--eval_path`: Path to the evaluation samples.

### Other arguments

*   `--hidden_sizes`: Number of layers and number of neurons per layer (default=[128, 64]).
*   `--lr`: Learning rate (default=0.001).
*   `--batch_size`: Batch size (default=32).
*   `--target_update_freq`: Update frequency of the target DQN (default=2000).
*   `--replay_size`: Replay buffer size (default=50000).
*   `--eval_epsilon`: Evaluation epsilon value (default=0.05).
*   `--final_epsilon`: Final epsilon value (default=0.05).
*   `--init_epsilon`: Initial epsilon value (default=1.0).
*   `--exploration_steps`: Number of steps to allow exploration (default=60000).
*   `--gamma`: Gamma value (default=0.99).
*   `--T_d`: Number of seconds from the first timestamp (default=30e9 i.e., 30 seconds).
*   `--T_window`: Size of the time window (default=1e8 i.e., 0.1 seconds).
*   `--base_reward`: Base reward (default=3).
*   `--checkpoint_freq`: Frequency of saving checkpoints in number of episodes (default: 100).
*   `--log_dir`: Directory to save tensorboard logs (default='runs').
*   `--trained_model_dir`: Directory to save/load the trained model (default: trained_models).
*   `--checkpoint_dir`: Directory to save checkpoints (default: checkpointed_models).
*   `--sample_paths_dir`: Directory to save sample paths YAML files (default: sample_paths).
*   `--duplicate_check`: How do you want to check for duplicates, hash based or distance based (default: distance).
*   `--distance_threshold`: The euclidean distance value to use for checking duplicates (default: 0.01).

## Training Time

Training for 1440 samples takes approximately 18-20 hours without a GPU.

## Citation

Please cite our research paper as follows:

```
[Citation information will be added once the paper is published]
```