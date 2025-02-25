import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import datetime
import os
import numpy as np
import pandas as pd
from state_computation import compute_state  # Import compute_state
from torch.utils.tensorboard import SummaryWriter
import glob
import yaml
import hashlib

class ReplayMemory:
    """
    A cyclic buffer to store and sample experience tuples for DQN training.  Handles duplicate state checking.
    """
    def __init__(self, capacity, s_dims, device="cpu", 
                 duplicate_check="distance", 
                 distance_threshold=0.01):
        """
        Initializes the ReplayMemory.

        Args:
            capacity (int): Maximum number of experiences to store.
            s_dims (tuple): Dimensions of the state space.
            device (str): Device to store the replay memory on ('cpu' or 'cuda').
            duplicate_check (str): Method for duplicate checking ('hash' or 'distance'). Default: 'distance'.
            distance_threshold (float): Threshold for distance-based duplicate checking. Default: 0.01.
        """
        self.capacity = capacity
        self.device = device
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0
        self.duplicate_check = duplicate_check
        self.distance_threshold = distance_threshold
        self.state_hashes = set()  # Set to store state hashes for hash-based duplicate checking
    
    def hash_state(self, state):
        """
        Calculates the SHA256 hash of the state vector.

        Args:
            state (np.ndarray): The state to hash.

        Returns:
            str: The hexadecimal representation of the hash.
        """
        return hashlib.sha256(state.tobytes()).hexdigest()

    def is_duplicate_distance(self, state):
        """
        Checks if the state is a duplicate based on Euclidean distance.

        Args:
            state (np.ndarray): The state to check.

        Returns:
            bool: True if the state is a duplicate, False otherwise.
        """
        for stored_state in self.s_buf[:self.size]:
            distance = np.linalg.norm(state - stored_state)
            if distance < self.distance_threshold:
                return True
        return False

    def store(self, s, a, next_s, r, done):
        """
        Stores an experience tuple in the replay memory, performing duplicate checking.

        Args:
            s (np.ndarray): State.
            a (int): Action.
            next_s (np.ndarray): Next state.
            r (float): Reward.
            done (bool): Done flag.
        """
        if self.duplicate_check == "hash":
            state_hash = self.hash_state(s)
            if state_hash in self.state_hashes:
                return  # Skip storing if duplicate hash
            self.state_hashes.add(state_hash)

        elif self.duplicate_check == "distance":
            if self.is_duplicate_distance(s):
                return  # Skip storing if duplicate based on distance
            
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        """
        Samples a batch of experiences from the replay memory for uniform sampling.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            list: A list of tensors representing the batch of experiences: states, actions, next states, rewards, dones.
        """
        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [
            self.s_buf[sample_idxs],
            self.a_buf[sample_idxs],
            self.next_s_buf[sample_idxs],
            self.r_buf[sample_idxs],
            self.done_buf[sample_idxs]
        ]
        return [torch.from_numpy(buf).to(self.device) for buf in batch]
    
    def priority_batch(self, batch_size):
        """
        Fetches the most recent batch of experiences from the replay memory for priority sampling.

        Args:
            batch_size (int): Number of experiences to fetch.

        Returns:
            list: A list of tensors representing the batch of experiences.
                  Returns None if batch_size exceeds the current buffer size.
        """
        end_idx = self.ptr  # The "pointer" indicates the *next* location to write
        start_idx = (end_idx - batch_size) % self.capacity # Calculate start index, handling wrap-around

        idxs = []
        for i in range(batch_size):
            idx = (start_idx + i) % self.capacity
            idxs.append(idx)

        batch = [
            self.s_buf[idxs],
            self.a_buf[idxs],
            self.next_s_buf[idxs],
            self.r_buf[idxs],
            self.done_buf[idxs]
        ]
        return [torch.from_numpy(buf).to(self.device) for buf in batch]

class DQN(nn.Module):
    """
    A deep Q-network (DQN) with a flexible number of fully connected layers.
    """
    def __init__(self, input_dim, layers, num_actions):
        """
        Initializes the DQN.

        Args:
            input_dim (int): Dimension of the input state.
            layers (list): List of integers representing the number of neurons in each hidden layer.
            num_actions (int): Number of possible actions.
        """
        super().__init__()

        # Basic linear layers that may not give the best accuracy
        # self.layers = nn.ModuleList([nn.Linear(input_dim, layers[0])])
        # for l in range(1, len(layers)):
        #     self.layers.append(nn.Linear(layers[l - 1], layers[l]))
        # self.out = nn.Linear(layers[-1], num_actions)

        # # Advanced conv and LSTM layers for better accuracy with detecting temporal patterns in ransomware data
        # # ToDo: Does not work currently because of not having a temporal dimension in our state vector
        # self.conv1d_layers = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1) for _ in range(input_dim)])
        # self.maxpool1d_layers = nn.ModuleList([nn.MaxPool1d(kernel_size=2) for _ in range(input_dim)])
        # self.lstm1 = nn.LSTM(input_dim, 256, batch_first=True)
        # self.lstm2 = nn.LSTM(256, 256, batch_first=True)
        # self.fc = nn.Linear(256, num_actions)

        # Enhanced Fully Connected Layers
        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() # Batch Normalization layers

        # Input layer
        self.layers.append(nn.Linear(input_dim, layers[0]))
        self.bn_layers.append(nn.BatchNorm1d(layers[0])) # Batch Norm for the first layer

        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(layers[i+1]))  # Batch Norm for each hidden layer
            self.layers.append(nn.Dropout(p=0.2)) # Dropout after each Batch Norm

        # Output layer
        self.out = nn.Linear(layers[-1], num_actions)

    def forward(self, x):
        """
        Performs a forward pass through the DQN.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output tensor of Q-values for each action.
        """
        # Basic RELU based forward pass
        # for layer in self.layers:
        #     x = F.relu(layer(x))
        # return self.out(x)
    
        # # Advanced LSTM based forward pass
        # # ToDo: Does not work currently because of not having a temporal dimension in our state vector
        # # Reshape input for Conv1D: (Batch, Features) -> (Batch, 1, Features)
        # x = x.unsqueeze(1)

        # # Apply Conv1D and MaxPooling for each feature
        # conv_outputs = []
        # for i in range(x.shape[2]):  # Iterate over features
        #     conv_out = self.conv1d_layers[i](x[:, :, i].unsqueeze(1))  # No additional activation after Conv1D
        #     pooled_out = self.maxpool1d_layers[i](conv_out)
        #     conv_outputs.append(pooled_out.squeeze(1))

        # x = torch.stack(conv_outputs, dim=2)  # Stack to combine outputs

        # # Apply LSTM layers
        # x, _ = self.lstm1(x)
        # x, _ = self.lstm2(x)

        # # Take output from last timestep and apply fully connected layer with softmax
        # x = x[:, -1, :]
        # x = F.softmax(self.fc(x), dim=1)  # Softmax for classification
        # return x

        # Enhanced Fully Connected Layers
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if i < len(self.bn_layers):
                x = self.bn_layers[i](x) # Input to BatchNorm is now (Batch, Features)
            x = F.relu(x) # Apply ReLU after Batch Norm (except for the last layer)
        x = F.softmax(self.out(x), dim=1) # Softmax activation
        return x

    def save_DQN(self, file_path):
        """
        Saves the DQN model to a file.

        Args:
            file_path (str): Path to the file to save the model.
        """
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path):
        """
        Loads the DQN model from a file.

        Args:
            file_path (str): Path to the file to load the model from.
        """
        self.load_state_dict(torch.load(file_path, weights_only=True))
    
    def get_feature_importance(self):
        """
        Calculates and returns feature importance scores based on the absolute values of the weights
        in the first layer of the DQN.  Averages the absolute weights across output neurons.

        Returns:
            np.ndarray: Feature importance scores.
        """
        with torch.no_grad():
            weights = self.layers[0].weight.abs().mean(dim=0)  # Take the absolute values and average across output neurons
        return weights.cpu().numpy()  # Convert to NumPy array

class Agent:
    """
    A Deep Q-Learning agent for ransomware detection and defense.
    """
    def __init__(self, state_dim, num_actions, lr=1, gamma=0.99,
                 batch_size=64, replay_size=50000, init_epsilon=1.0, final_epsilon=0.05,
                 exploration_steps=60000, target_update_freq=2000, hidden_sizes=[128, 128, 128], 
                 log_dir="runs", duplicate_check="distance", distance_threshold=0.01,
                 beta=0.01, rho=0.95, eps=1e-6, sampling_type="uniform"):
        """
        Initializes the Agent.

        Args:
            state_dim (int): Dimension of the state space.
            num_actions (int): Number of possible actions.
            lr (float): Learning rate. Default: 1.
            gamma (float): Discount factor. Default: 0.99.
            batch_size (int): Batch size for training. Default: 64.
            replay_size (int): Replay memory size. Default: 50000.
            init_epsilon (float): Initial epsilon value. Default: 1.0.
            final_epsilon (float): Final epsilon value. Default: 0.05.
            exploration_steps (int): Number of steps to allow exploration. Default: 60000.
            target_update_freq (int): Target network update frequency. Default: 2000.
            hidden_sizes (list): List of hidden layer sizes for the DQN. Default: [128, 128, 128].
            log_dir (str): Directory to save TensorBoard logs. Default: "runs".
            duplicate_check (str): Method for duplicate state checking ('hash' or 'distance'). Default: 'distance'.
            distance_threshold (float): Threshold for distance-based duplicate checking. Default: 0.01.
            beta (float): Decay rate for reward based on engagement length. Default: 0.01.
            rho (float): Smoothing constant for Adadelta optimizer. Default: 0.95.
            eps (float): Small value added to denominator for numerical stability in Adadelta. Default: 1e-6.
            sampling_type (str): Type of sampling to use for replay memory ('uniform' or 'priority'). Default: 'uniform'.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = num_actions
        self.dqn = DQN(state_dim, hidden_sizes, self.num_actions).to(self.device)
        self.target_dqn = DQN(state_dim, hidden_sizes, self.num_actions).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        # self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.optimizer = optim.Adadelta(self.dqn.parameters(), lr=lr, rho=rho, eps=eps)
        self.gamma = gamma
        self.batch_size = batch_size
        self.sampling_type = sampling_type
        self.replay = ReplayMemory(replay_size, (state_dim,), self.device, duplicate_check, distance_threshold)
        self.exploration_steps = exploration_steps
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(self.init_epsilon, self.final_epsilon, self.exploration_steps)
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        # self.loss_fn = nn.SmoothL1Loss() # typically used for regression tasks, where the goal is to predict a continuous value.
        self.loss_fn = nn.CrossEntropyLoss() # specifically designed for multi-class classification problems where the output is a probability distribution.
        self.writer = SummaryWriter(log_dir=log_dir)
        self.last_episode = 0
        self.beta = beta
        self.curr_engagement_length = 0
        # Initialize lists to store results for metric calculation
        self.y_true = []
        self.y_pred = []
        self.episode_rewards = []
        self.goal_completion = []
        self.loss = []
        self.mean_v = []
    
    # DataFrame loading with timestamp calculation and column names
    def load_data(self, file_prefix):
        """Loads dataframes from CSV files, adds a timestamp column in nanoseconds, and handles missing files.
        
        Args:
            file_prefix (str): The prefix for the CSV file names (including path), e.g., 'path/to/ata' or 'path/to/mem'.

        Returns:
            tuple: A tuple of Pandas DataFrames (df_read, df_write) for ATA logs, or 
                   (df_read, df_write, df_readwrite, df_exec) for memory logs.  
                   DataFrames have column names and 'ts' in nanoseconds.  
                   Returns None if there is an error during loading or any required file is missing.
        """
        try:
            if 'mem' in file_prefix:
                df_read = pd.read_csv(f"{file_prefix}_read.csv", header=None, names=['ts_s', 'ts_ns', 'GPA', 'size', 'entropy', 'type']) if os.path.exists(f"{file_prefix}_read.csv") else None
                df_write = pd.read_csv(f"{file_prefix}_write.csv", header=None, names=['ts_s', 'ts_ns', 'GPA', 'size', 'entropy', 'type']) if os.path.exists(f"{file_prefix}_write.csv") else None
                df_readwrite = pd.read_csv(f"{file_prefix}_readwrite.csv", header=None, names=['ts_s', 'ts_ns', 'GPA', 'size', 'entropy', 'type']) if os.path.exists(f"{file_prefix}_readwrite.csv") else None
                df_exec = pd.read_csv(f"{file_prefix}_exec.csv", header=None, names=['ts_s', 'ts_ns', 'GPA', 'size', 'entropy', 'type']) if os.path.exists(f"{file_prefix}_exec.csv") else None
            else:
                df_read = pd.read_csv(f"{file_prefix}_read.csv", header=None, names=['ts_s', 'ts_ns', 'LBA', 'size', 'entropy', 'type']) if os.path.exists(f"{file_prefix}_read.csv") else None
                df_write = pd.read_csv(f"{file_prefix}_write.csv", header=None, names=['ts_s', 'ts_ns', 'LBA', 'size', 'entropy', 'type']) if os.path.exists(f"{file_prefix}_write.csv") else None
            if df_read is not None and df_write is not None:
                df_read['ts'] = df_read['ts_s'] * 1e9 + df_read['ts_ns']
                df_write['ts'] = df_write['ts_s'] * 1e9 + df_write['ts_ns']            
            if 'mem' in file_prefix:
                if df_readwrite is not None and df_exec is not None:
                    df_readwrite['ts'] = df_readwrite['ts_s'] * 1e9 + df_readwrite['ts_ns']
                    df_exec['ts'] = df_exec['ts_s'] * 1e9 + df_exec['ts_ns']
                return df_read, df_write, df_readwrite, df_exec
            else:
                return df_read, df_write
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    
    def get_epsilon(self):
        """
        Gets the current epsilon value based on the exploration schedule.

        Returns:
            float: The current epsilon value.
        """
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def select_action(self, state, epsilon):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state.
            epsilon (float): Exploration probability.

        Returns:
            int: The selected action (0 or 1).
        """
        if random.random() > epsilon: # Exploit: use the model to make the decision
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            state = state.unsqueeze(0)  # Add batch dimension
            self.dqn.eval() # We need to set the model to eval mode so that batch normalization does not look for multiple batches
            with torch.no_grad():
                q_values = self.dqn(state) # Note: We do not need to call the forward function because PyTorch takes care of it inherently
            self.dqn.train()
            action = torch.argmax(q_values).item()
        else:
            action = random.randint(0, self.num_actions - 1) # Explore: choose a random action
        return action

    def compute_reward(self, action, base_reward, cost, is_malicious):
        """
        Computes the reward based on the action, cost, and whether the sample is malicious.
        Reward is adjusted based on the current engagement length using exponential decay.

        Args:
            action (int): Action taken (0 or 1).
            base_reward (float): Base reward value.
            cost (list): List of costs for each action.
            is_malicious (bool): True if the sample is malicious.

        Returns:
            float: The computed reward.
        """
        # Naive Rewards.
        # defaults: base_reward = 3, cost = [1, 2]
        # if action == 0 and not is_malicious:
        #     reward = base_reward - cost[action] # =2 if action is continue and trace is benign, reward of x
        # elif action == 1 and is_malicious: # tested: 1.5
        #     # testing with 2 to equalize how much agent weighs both actions
        #     reward = 2 * (base_reward - cost[action]) # =2 if action is defend and trace is malicious, reward of 1.5x
        # elif action == 1 and not is_malicious: # tested: -1
        #     # testing with -3 to equalize how much the agent weighs both actions
        #     reward = -3 * (base_reward - cost[action]) # =-3 if action is defend and trace is benign, reward of -x
        # elif action == 0 and is_malicious: # tested: -1.5
        #     # testing with -1.5 to equalize how much the agent weighs both actions
        #     reward = -1.5 * (base_reward - cost[action]) # =-3 if action is continue and trace is malicious, reward of -2x

        # Reward based on engagement length
        reward = (base_reward - (2 * base_reward * (abs(action - int(is_malicious))))) * np.exp(self.beta * self.curr_engagement_length)
        return reward

    def optimize(self):
        """
        Performs a single optimization step for the DQN.

        Samples a batch from replay memory, computes the loss using Huber loss (SmoothL1Loss),
        and updates the DQN's parameters using the Adadelta optimizer.
        Periodically updates the target DQN by copying the DQN's weights.

        Returns:
            tuple: (float, float) Loss value and mean Q-value.
                   Returns (0, 0) if the replay memory has fewer samples than the batch size.
        """
        if self.replay.size < self.batch_size:
            return 0, 0  # Not enough samples for a batch
        
        if self.sampling_type == "uniform":
            s_batch, a_batch, next_s_batch, r_batch, d_batch = self.replay.sample_batch(self.batch_size)
        elif self.sampling_type == "priority":
            s_batch, a_batch, next_s_batch, r_batch, d_batch = self.replay.priority_batch(self.batch_size)
        else:
            raise ValueError("Invalid sampling type. Please choose 'uniform' or 'priority'.")

        # # **RESHAPE s_batch and next_s_batch HERE** (only for the advanced CNN + LSTM approach)
        # s_batch = s_batch.unsqueeze(1)       # Add temporal dimension for Conv1d and LSTM
        # next_s_batch = next_s_batch.unsqueeze(1)  # Do the same for next_s_batch

        # Compute Q-values for current states and actions
        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        # Compute target Q-values using target network and Bellman equation
        with torch.no_grad():
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_val = target_q_val_raw.max(1)[0]
            target = r_batch + self.gamma * (1 - d_batch) * target_q_val

        # Compute loss using Huber loss
        loss = self.loss_fn(q_vals, target)

        # Optimize the DQN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        # Calculate mean Q-value
        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()

        return loss.item(), mean_v

    def train(self, train_paths, num_episodes, base_reward, cost, max_engagement_length=30, T_max=30e9, T_window=1e8, checkpoint_freq=100, checkpoint_dir="checkpointed_models"):
        """
        Trains the DQN agent.

        Iterates through episodes, loads data, selects actions, computes rewards,
        and performs optimization steps. Also saves checkpoints periodically.

        Args:
            train_paths (list): List of tuples: (ata_path, mem_path, is_malicious).
            num_episodes (int): Total number of training episodes.
            base_reward (float): Base reward value.
            cost (list): List of costs for each action.
            max_engagement_length (int): Maximum consecutive steps with positive reward to consider a goal reached.
            T_max (float): Time duration for an episode in nanoseconds.
            T_window (float): Time window size in nanoseconds.
            checkpoint_freq (int): Save checkpoint every `checkpoint_freq` episodes.
            checkpoint_dir (str): Directory to save checkpoints.
        """

        # Check for existing checkpoints to resume training
        last_checkpoint = self.get_last_checkpoint(checkpoint_dir)
        if last_checkpoint:
            self.load_checkpoint(last_checkpoint)
            start_episode = self.last_episode + 1  # Resume from the next episode
            print(f"Resuming training from checkpoint: {last_checkpoint}, episode: {start_episode}")
            
            train_paths_path = os.path.join(checkpoint_dir, "train_paths.yaml")
            if os.path.exists(train_paths_path):
                with open(train_paths_path, 'r') as f:
                    train_paths = yaml.safe_load(f)
                print(f"Loaded train_paths from checkpoint directory.")
            else:
                print("train_paths.yaml not found in checkpoint directory. Using the provided train_paths.")
        else:
            start_episode = 0
            # Save train_paths once at the beginning
            train_paths_path = os.path.join(checkpoint_dir, "train_paths.yaml")
            if not os.path.exists(train_paths_path):
                with open(train_paths_path, 'w') as f:
                    yaml.dump(train_paths, f)
                print(f"Saved train_paths to {train_paths_path}")

        for episode in range(start_episode, num_episodes):
            # Get the sample for this episode
            ata_path, mem_path, is_malicious = train_paths[episode]

            # Load DataFrames for the selected sample
            df_ata_read, df_ata_write = self.load_data(ata_path)
            df_mem_read, df_mem_write, df_mem_readwrite, df_mem_exec = self.load_data(mem_path)
            
            # Check if any of the DataFrames are None
            if df_ata_read is None or df_ata_write is None or df_mem_read is None or \
               df_mem_write is None or df_mem_readwrite is None or df_mem_exec is None:
                print(f"Skipping episode {episode + 1} due to data loading error.")
                continue  # Skip to the next episode

            t = min(df_ata_read['ts'].min(), df_ata_write['ts'].min(),
                    df_mem_read['ts'].min(), df_mem_write['ts'].min(),
                    df_mem_readwrite['ts'].min(), df_mem_exec['ts'].min())
            
            # To avoid situations where we start fetching null states
            t_max = min(df_ata_read['ts'].max(), df_ata_write['ts'].max(), 
                        df_mem_read['ts'].max(), df_mem_write['ts'].max(), 
                        df_mem_readwrite['ts'].max(), df_mem_exec['ts'].max())
            
            t_max = t + T_max if (t + T_max) < t_max else t_max
            done = False
            episode_return = 0
            max_episode_return = 0
            self.curr_engagement_length = 0
            steps_done = 0
            max_steps = (t_max - t) / T_window

            while t <= (t_max - T_window) and not done:
                state = compute_state(df_ata_read, df_ata_write, df_mem_read,
                                     df_mem_write, df_mem_readwrite, df_mem_exec, t, T_window)
                
                epsilon = self.get_epsilon()
                
                action = self.select_action(state, epsilon)  

                reward = self.compute_reward(action, base_reward, cost, is_malicious)

                next_t = t + T_window
                next_state = compute_state(df_ata_read, df_ata_write, df_mem_read,
                                         df_mem_write, df_mem_readwrite, df_mem_exec, next_t, T_window)
                
                # This is a goal condition per episode.
                # We essentially want to stop when the number of steps we get consecutive positive rewards,
                # reaches a maximum engagement length.
                # 
                # This is representative of layered security defenses where,
                # the proactive defense layer (1st layer) has to engage an unknown application until the
                # lower layers like the detectors can confidently make a decision about the maliciousness
                # of the unknown application.
                if reward > 0:
                    self.curr_engagement_length += 1
                    episode_return += reward
                    if self.curr_engagement_length == max_engagement_length:
                        done = True
                        max_episode_return = episode_return
                        
                elif reward < 0: # Reset engagement length if reward is negative
                    if episode_return > max_episode_return:
                        max_episode_return = episode_return
                    self.curr_engagement_length = 0
                    episode_return = 0

                self.replay.store(state, action, next_state, reward, done)

                loss, mean_v = self.optimize()

                self.writer.add_scalar("Loss/train", loss, self.steps_done) # log the loss
                self.writer.add_scalar("Mean V/train", mean_v, self.steps_done) # log mean v

                self.steps_done += 1
                steps_done += 1
                t = next_t

                # Since we only have binary actions for binary ground truth
                # If we expand the actions this would change
                self.y_true.append(int(is_malicious))
                self.y_pred.append(action)

                # Track the loss and mean V values during training
                self.loss.append((self.steps_done, loss))
                self.mean_v.append((self.steps_done, mean_v))

            completion_point = steps_done / max_steps
            
            self.goal_completion.append((done, completion_point, is_malicious))

            self.episode_rewards.append((episode + 1, max_episode_return, is_malicious))
            
            if (episode + 1) % checkpoint_freq == 0:  # Check if it's time to save a checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode + 1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
                self.save_checkpoint(checkpoint_path, episode)
                print(f"Checkpoint saved at episode {episode + 1} to {checkpoint_path}")

            self.writer.add_scalar("Training Reward", max_episode_return, episode)  # Log all episode rewards
            if is_malicious:
                self.writer.add_scalar("Training Reward (malware samples)", max_episode_return, episode)  # Log rewards for malware samples
            if not is_malicious:
                self.writer.add_scalar("Training Reward (benign samples)", max_episode_return, episode)  # Log reward for benign samples

            print(f"Training Episode {episode + 1}: malware={is_malicious}, return={max_episode_return}, steps={self.steps_done}, epsilon={epsilon}, goal reached={done}")

    def evaluate(self, eval_paths, num_episodes, base_reward, cost, max_engagement_length=30, eval_epsilon=0.05, T_max=30e9, T_window=1e8):
        """
        Evaluates the trained DQN agent.

        Iterates through episodes, loads data, selects actions, and computes rewards
        without updating the agent.

        Args:
            eval_paths (list): List of tuples: (ata_path, mem_path, is_malicious).
            num_episodes (int): Total number of evaluation episodes.
            base_reward (float): Base reward value.
            cost (list): List of costs for each action.
            max_engagement_length (int): Maximum consecutive steps with positive reward to consider a goal reached.
            eval_epsilon (float): Epsilon value for action selection during evaluation.
            T_max (float): Time duration for an episode in nanoseconds.
            T_window (float): Time window size in nanoseconds.
        """

        for episode in range(num_episodes):
            # Get the sample for this episode
            ata_path, mem_path, is_malicious = eval_paths[episode]

            # Load DataFrames for the selected sample
            df_ata_read, df_ata_write = self.load_data(ata_path)
            df_mem_read, df_mem_write, df_mem_readwrite, df_mem_exec = self.load_data(mem_path)

            # Check if any of the DataFrames are None
            if df_ata_read is None or df_ata_write is None or df_mem_read is None or \
               df_mem_write is None or df_mem_readwrite is None or df_mem_exec is None:
                print(f"Skipping episode {episode + 1} due to data loading error.")
                continue  # Skip to the next episode

            t = min(df_ata_read['ts'].min(), df_ata_write['ts'].min(),
                    df_mem_read['ts'].min(), df_mem_write['ts'].min(),
                    df_mem_readwrite['ts'].min(), df_mem_exec['ts'].min())
            
            # To avoid situations where we start fetching null states
            t_max = min(df_ata_read['ts'].max(), df_ata_write['ts'].max(), 
                        df_mem_read['ts'].max(), df_mem_write['ts'].max(), 
                        df_mem_readwrite['ts'].max(), df_mem_exec['ts'].max())
            
            t_max = t + T_max if (t + T_max) < t_max else t_max
            done = False
            episode_return = 0
            max_episode_return = 0
            self.curr_engagement_length = 0
            steps_done = 0
            max_steps = (t_max - t) / T_window

            while t <= (t_max - T_window) and not done:
                state = compute_state(df_ata_read, df_ata_write, df_mem_read,
                                     df_mem_write, df_mem_readwrite, df_mem_exec, t, T_window)
                
                action = self.select_action(state, eval_epsilon)  

                reward = self.compute_reward(action, base_reward, cost, is_malicious)

                next_t = t + T_window

                self.steps_done += 1
                steps_done += 1
                episode_return += reward
                t = next_t

                # Since we only have binary actions for binary ground truth
                # If we expand the actions this would change
                self.y_true.append(int(is_malicious))
                self.y_pred.append(action)

                # This is a goal condition per episode.
                # We essentially want to stop when the number of steps we get consecutive positive rewards,
                # reaches a maximum engagement length.
                # 
                # This is representative of layered security defenses where,
                # the proactive defense layer (1st layer) has to engage an unknown application until the
                # lower layers like the detectors can confidently make a decision about the maliciousness
                # of the unknown application.
                if reward > 0:
                    self.curr_engagement_length += 1
                    episode_return += reward
                    if self.curr_engagement_length == max_engagement_length:
                        done = True
                        max_episode_return = episode_return
                        
                elif reward < 0: # Reset engagement length and return if reward is negative
                    if episode_return > max_episode_return:
                        max_episode_return = episode_return
                    self.curr_engagement_length = 0
                    episode_return = 0
            
            completion_point = steps_done / max_steps
            
            self.goal_completion.append((done, completion_point, is_malicious))

            self.episode_rewards.append((episode + 1, max_episode_return, is_malicious))

            self.writer.add_scalar("Evaluation Reward", max_episode_return, episode)  # Log all episode rewards
            if is_malicious:
                self.writer.add_scalar("Evaluation Reward (malware samples)", max_episode_return, episode)  # Log rewards for malware samples
            if not is_malicious:
                self.writer.add_scalar("Evaluation Reward (benign samples)", max_episode_return, episode)  # Log reward for benign samples

            print(f"Evaluation Episode {episode + 1}: malware={is_malicious}, return={max_episode_return}, steps={self.steps_done}, goal reached={done}")

    def save(self, save_path):
        """
        Saves the DQN model weights to a file.

        Args:
            save_path (str): The path to save the model to.
        """
        self.dqn.save_DQN(save_path)

    def load(self, load_path):
        """
        Loads the DQN model weights from a file.

        Args:
            load_path (str): Path to load the model weights from.
        """
        self.dqn.load_DQN(load_path)

    def save_checkpoint(self, save_path, episode):
        """
        Saves a checkpoint of the agent's state, including model weights, optimizer state,
        and current episode number.

        Args:
            save_path (str): The path to save the checkpoint to.
            episode (int): The current episode number.
        """
        torch.save({
            'steps_done': self.steps_done,
            'model_state_dict': self.dqn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'last_episode': episode,  # Save the last episode number
            # Add any other objects you want to save
        }, save_path)

    def load_checkpoint(self, load_path):
        """
        Loads a checkpoint of the agent's state, including model weights, optimizer state,
        and current episode number.

        Args:
            load_path (str): The path to load the checkpoint from.
        """
        checkpoint = torch.load(load_path, weights_only=True)
        self.steps_done = checkpoint['steps_done']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dqn.load_state_dict(checkpoint['model_state_dict'])
        self.last_episode = checkpoint.get('last_episode', 0)  # Load last episode, default to 0 if not found
        # Load any other objects you want to load

    def get_last_checkpoint(self, checkpoint_dir):
        """
        Finds the last saved checkpoint in the given directory.

        Args:
            checkpoint_dir (str): The directory where checkpoints are saved.

        Returns:
            str or None: The path to the last checkpoint, or None if no checkpoints were found.
        """
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
        if checkpoint_files:
            # Find the checkpoint with the highest episode number
            checkpoint_files.sort(key=lambda x: int(x.split('_')[2]))
            return checkpoint_files[-1]
        return None
    
    def delete_checkpoints(self, checkpoint_dir):
        """
        Deletes all checkpoint files and the train_paths.yaml file in the given directory.

        Args:
            checkpoint_dir (str): The directory to delete checkpoints from.
        """
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith(".pth") or filename.endswith(".yaml"): # delete pth and yaml
                file_path = os.path.join(checkpoint_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting checkpoint: {e}")