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
    A cyclic buffer to store and sample experience tuples for DQN training.

    Attributes:
        capacity (int): Maximum number of experiences to store.
        s_buf (np.ndarray): Buffer to store states.
        a_buf (np.ndarray): Buffer to store actions.
        next_s_buf (np.ndarray): Buffer to store next states.
        r_buf (np.ndarray): Buffer to store rewards.
        done_buf (np.ndarray): Buffer to store done flags.
        ptr (int): Pointer to the current position in the buffer.
        size (int): Current number of experiences stored in the buffer.
    """
    def __init__(self, capacity, s_dims, device="cpu", 
                 duplicate_check="distance",  # Add duplicate_check parameter
                 distance_threshold=0.01):  # Add distance_threshold parameter):
        """
        Initializes the ReplayMemory.

        Args:
            capacity (int): Maximum number of experiences to store.
            s_dims (tuple): Dimensions of the state space.
            device (str): Device to store the replay memory on ('cpu' or 'cuda').
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
        self.state_hashes = set()  # Set to store state hashes
    
    def hash_state(self, state):
        """Calculates the hash of the state vector."""
        return hashlib.sha256(state.tobytes()).hexdigest()

    def is_duplicate_distance(self, state):
        """Checks if the state is a duplicate based on distance."""
        for stored_state in self.s_buf[:self.size]:
            distance = np.linalg.norm(state - stored_state)
            if distance < self.distance_threshold:
                return True
        return False

    def store(self, s, a, next_s, r, done):
        """
        Stores an experience tuple in the replay memory.

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
        Samples a batch of experiences from the replay memory.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            list: A list of tensors representing the batch of experiences.
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

class DQN(nn.Module):
    """
    A deep Q-network (DQN) with a flexible number of fully connected layers.

    Attributes:
        layers (nn.ModuleList): List of fully connected layers.
        out (nn.Linear): Output layer.
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
        self.layers = nn.ModuleList([nn.Linear(input_dim, layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l - 1], layers[l]))
        self.out = nn.Linear(layers[-1], num_actions)

    def forward(self, x):
        """
        Performs a forward pass through the DQN.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output tensor of Q-values for each action.
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)

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

class Agent:
    """
    A Deep Q-Learning agent.

    Attributes:
        device (torch.device): Device to run the agent on ('cpu' or 'cuda').
        dqn (DQN): Main DQN model.
        target_dqn (DQN): Target DQN model.
        optimizer (torch.optim.Optimizer): Optimizer for training the DQN.
        gamma (float): Discount factor.
        batch_size (int): Batch size for training.
        replay (ReplayMemory): Replay memory.
        exploration_steps (int): Number of exploration steps.
        init_epsilon (float): Initial epsilon value.
        final_epsilon (float): Final epsilon value.
        epsilon_schedule (np.ndarray): Epsilon schedule.
        target_update_freq (int): Target network update frequency.
        steps_done (int): Number of steps taken.
        loss_fn (torch.nn.Module): Loss function.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard SummaryWriter.
    """
    def __init__(self, state_dim, num_actions, lr=0.001, gamma=0.99,
                 batch_size=32, replay_size=10000,init_epsilon=1.0, final_epsilon=0.05,
                 exploration_steps=100, target_update_freq=10, hidden_sizes=[64, 64], 
                 log_dir="runs", duplicate_check="distance", distance_threshold=0.01):
        """
        Initializes the Agent.

        Args:
            state_dim (int): Dimension of the state space.
            num_actions (int): Number of possible actions.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            batch_size (int): Batch size for training.
            replay_size (int): Replay memory size.
            init_epsilon (float): Initial epsilon value.
            final_epsilon (float): Final epsilon value.
            exploration_steps (int): Number of exploration steps.
            target_update_freq (int): Target network update frequency.
            hidden_sizes (list): List of hidden layer sizes for the DQN.
            log_dir (str): Directory to save TensorBoard logs.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(state_dim, hidden_sizes, num_actions).to(self.device)
        self.target_dqn = DQN(state_dim, hidden_sizes, num_actions).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay = ReplayMemory(replay_size, (state_dim,), self.device, duplicate_check, distance_threshold)
        self.exploration_steps = exploration_steps
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(self.init_epsilon, self.final_epsilon, self.exploration_steps)
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        self.loss_fn = nn.SmoothL1Loss()
        self.writer = SummaryWriter(log_dir=log_dir)
    
    # DataFrame loading with timestamp calculation and column names
    def load_data(self, file_prefix):
        """
        Loads dataframes from CSV files and creates a timestamp column.

        Args:
            file_prefix (str): The prefix for the file names (e.g., 'ata' or 'mem').

        Returns:
            tuple: A tuple of Pandas DataFrames with column names and 'ts' in nanoseconds.
                   Returns None if there is an error loading the data.
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
        Returns the epsilon value for the current step, decaying linearly 
        from init_epsilon to final_epsilon over exploration_steps.

        Returns:
            float: Current epsilon value.
        """
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def select_action(self, state, epsilon):
        """
        Selects an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state of the environment.
            epsilon (float): Probability of taking a random action.

        Returns:
            int: Selected action.
        """
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.dqn(state)
            action = torch.argmax(q_values).item()
        else:
            action = random.randint(0, 1)
        return action

    def compute_reward(self, action, base_reward, cost, is_malicious):
        """
        Computes the reward based on the action taken and whether the trace is malicious.

        Args:
            action (int): Action taken by the agent (0 or 1).
            base_reward (int): Base reward value.
            cost (list): List of costs for each action.
            is_malicious (bool): Whether the trace is malicious.

        Returns:
            int: Computed reward.
        """
        # ToDo: Might have to re-caliberate this because we are currently not rewarding the model
        # enough for picking defend as an action.
        if action == 0 and not is_malicious:
            reward = base_reward - cost[action] # if action is continue and trace is benign, reward of x
        elif action == 1 and is_malicious:
            reward = 1.5 * (base_reward - cost[action]) # if action is defend and trace is malicious, reward of 1.5x
        elif action == 1 and not is_malicious:
            reward = -1 * (base_reward - cost[action]) # if action is defend and trace is benign, reward of -x
        elif action == 0 and is_malicious:
            # testing with -1.5 so equalize how much the agent weighs both actions
            reward = -1.5 * (base_reward - cost[action]) # if action is continue and trace is malicious, reward of -2x
        return reward

    def optimize(self):
        """
        Performs a single optimization step on the DQN.

        Samples a batch from replay memory, calculates the loss, and updates the DQN.
        Also updates the target DQN periodically.

        Returns:
            tuple: (float, float) Loss value and mean Q-value.
                   Returns (0, 0) if there are not enough samples in the replay memory.
        """
        if self.replay.size < self.batch_size:
            return 0, 0

        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        with torch.no_grad():
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_val = target_q_val_raw.max(1)[0]
            target = r_batch + self.gamma * (1 - d_batch) * target_q_val

        loss = self.loss_fn(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()

        self.writer.add_scalar("Loss/train", loss, self.steps_done) # log the loss
        self.writer.add_scalar("Mean V/train", mean_v, self.steps_done) # log mean v

        return loss.item(), mean_v

    def train(self, train_paths, num_episodes, base_reward, cost, T_d=30e9, T_window=1e8, checkpoint_freq=100, checkpoint_dir="checkpointed_models"):
        """
        Trains the DQN agent.

        Iterates through episodes, loads data, selects actions, computes rewards,
        and performs optimization steps. Also saves checkpoints periodically.

        Args:
            train_paths (list): List of tuples containing paths to training data and labels.
            num_episodes (int): Total number of episodes to train for.
            base_reward (int): Base reward value.
            cost (list): List of costs for each action.
            T_d (float): Time duration in nanoseconds (default: 30e9, i.e., 30 seconds).
            T_window (float): Time window in nanoseconds (default: 1e8, i.e., 0.1 seconds).
            checkpoint_freq (int): Frequency of saving checkpoints in number of episodes (default: 100).
            checkpoint_dir (str): Directory to save checkpoints (default: "checkpointed_models").

        Returns:
            float: Total reward accumulated during training.
        """
        total_reward = 0

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
            t_d = t + T_d
            # print(f"t={t}, T_d={t_d}")
            done = False
            episode_return = 0

            while t <= (t_d - T_window) and not done:
                state = compute_state(df_ata_read, df_ata_write, df_mem_read,
                                     df_mem_write, df_mem_readwrite, df_mem_exec, t, T_window)
                
                epsilon = self.get_epsilon()
                
                action = self.select_action(state, epsilon)  

                reward = self.compute_reward(action, base_reward, cost, is_malicious)

                next_t = t + T_window
                next_state = compute_state(df_ata_read, df_ata_write, df_mem_read,
                                         df_mem_write, df_mem_readwrite, df_mem_exec, next_t, T_window)
                done = next_t >= t_d

                self.replay.store(state, action, next_state, reward, done)
                
                loss, mean_v = self.optimize()

                self.steps_done += 1
                episode_return += reward
                t = next_t
            
            if (episode + 1) % checkpoint_freq == 0:  # Check if it's time to save a checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode+1}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
                self.save_checkpoint(checkpoint_path, episode)
                print(f"Checkpoint saved at episode {episode+1} to {checkpoint_path}")

            total_reward += episode_return
            self.writer.add_scalar("Training Reward", episode_return, episode)  # Log all episode rewards
            if is_malicious:
                self.writer.add_scalar("Training Reward (malware samples)", episode_return, episode)  # Log rewards for malware samples
            if not is_malicious:
                self.writer.add_scalar("Training Reward (benign samples)", episode_return, episode)  # Log reward for benign samples

            print(f"Training Episode {episode + 1}: malware={is_malicious}, return={episode_return}, steps={self.steps_done}, epsilon={epsilon}")
        
        return total_reward

    def evaluate(self, eval_paths, num_episodes, base_reward, cost, eval_epsilon=0.05, T_d=30e9, T_window=1e8):
        """
        Evaluates the trained DQN agent.

        Iterates through episodes, loads data, selects actions, and computes rewards
        without updating the agent.

        Args:
            eval_paths (list): List of tuples containing paths to evaluation data and labels.
            num_episodes (int): Total number of episodes to evaluate for.
            base_reward (int): Base reward value.
            cost (list): List of costs for each action.
            eval_epsilon (float): Epsilon value for evaluation (default: 0.05).
            T_d (float): Time duration in nanoseconds (default: 30e9, i.e., 30 seconds).
            T_window (float): Time window in nanoseconds (default: 1e8, i.e., 0.1 seconds).

        Returns:
            float: Total reward accumulated during evaluation.
        """
        total_reward = 0
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
            t_d = t + T_d
            done = False
            episode_return = 0

            while t <= (t_d - T_window) and not done:
                state = compute_state(df_ata_read, df_ata_write, df_mem_read,
                                     df_mem_write, df_mem_readwrite, df_mem_exec, t, T_window)
                
                action = self.select_action(state, eval_epsilon)  

                reward = self.compute_reward(action, base_reward, cost, is_malicious)

                next_t = t + T_window
                done = next_t >= t_d

                self.steps_done += 1
                episode_return += reward
                t = next_t

                self.writer.add_scalar("Reward/eval", episode_return, self.steps_done)  # Log episode return
            
            total_reward += episode_return

            self.writer.add_scalar("Evaluation Reward", episode_return, episode)  # Log all episode rewards
            if is_malicious:
                self.writer.add_scalar("Evaluation Reward (malware samples)", episode_return, episode)  # Log rewards for malware samples
            if not is_malicious:
                self.writer.add_scalar("Evaluation Reward (benign samples)", episode_return, episode)  # Log reward for benign samples

            print(f"Evaluation Episode {episode + 1}: malware={is_malicious}, return={episode_return}, steps={self.steps_done}")
        
        return total_reward
    
    def save(self, save_path):
        """
        Saves the DQN model weights to a file.

        Args:
            save_path (str): Path to save the model weights.
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
            save_path (str): Path to save the checkpoint.
            episode (int): Current episode number.
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
            load_path (str): Path to load the checkpoint from.
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
            checkpoint_dir (str): Directory to search for checkpoints.

        Returns:
            str: Path to the last checkpoint file, or None if no checkpoints are found.
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
            checkpoint_dir (str): Directory to delete checkpoints from.
        """
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith(".pth") or filename.endswith(".yaml"):
                file_path = os.path.join(checkpoint_dir, filename)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting checkpoint: {e}")