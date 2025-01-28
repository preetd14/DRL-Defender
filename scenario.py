import os
import yaml
import numpy as np
# import random

class Scenario:
    """
    Generates and saves sample paths for different scenarios.
    """

    def __init__(self, name, root_dir, mode, split_ratio=0.75):
        """
        Initializes the Scenario object.

        Args:
            name (str): Name of the scenario.
            root_dir (str): Root directory of the samples.
            split_ratio (float): Ratio of samples to use for training (default 0.75).
            mode (str): Mode of operation ('Training' or 'Evaluation').
        """
        self.name = name
        self.root_dir = root_dir
        self.sample_paths = []
        self.mode = mode
        self.split_ratio = split_ratio

    def generate_paths(self):
        """
        Generates sample paths by walking through the directory structure.

        Walks through the root directory, identifies samples based on their type
        (malicious or benign), and stores the paths in the sample_paths list.
        """
        for machine_type in os.listdir(self.root_dir):
            machine_path = os.path.join(self.root_dir, machine_type)
            if os.path.isdir(machine_path):
                for memory_type in os.listdir(machine_path):
                    memory_path = os.path.join(machine_path, memory_type)
                    if os.path.isdir(memory_path):
                        for app_type in os.listdir(memory_path):
                            app_path = os.path.join(memory_path, app_type)
                            if os.path.isdir(app_path):
                                for sample_dir in os.listdir(app_path):
                                    sample_path = os.path.join(app_path, sample_dir)
                                    if os.path.isdir(sample_path):
                                        ata_path = os.path.join(sample_path, "ata")
                                        mem_path = os.path.join(sample_path, "mem")
                                        is_malicious = app_type in ("WannaCry", "Ryuk", "REvil", "LockBit", "Darkside", "Conti")
                                        self.sample_paths.append((ata_path, mem_path, is_malicious))

    def split_paths(self):
        """
        Splits the sample paths into training and evaluation sets based on the mode.

        Shuffles and splits the sample paths based on the specified split ratio and mode.
        Ensures a balanced split of malicious and benign samples.

        Returns:
            list: The training or evaluation sample paths based on the mode.
        """
        # Shuffle the sample paths to randomize the split
        # random.shuffle(self.sample_paths)

        # Separate malicious and benign samples
        malicious_samples = [sample for sample in self.sample_paths if sample[2]]
        benign_samples = [sample for sample in self.sample_paths if not sample[2]]

        # Calculate the split index for each type of sample
        split_index_malicious = int(len(malicious_samples) * self.split_ratio)
        split_index_benign = int(len(benign_samples) * self.split_ratio)

        if self.mode == "Training":
            # Split and combine malicious and benign samples for training
            return malicious_samples[:split_index_malicious] + benign_samples[:split_index_benign]
        elif self.mode == "Evaluation":
            # Split and combine malicious and benign samples for evaluation
            return malicious_samples[split_index_malicious:] + benign_samples[split_index_benign:]

    def save_to_yaml(self, file_path):
        """
        Saves the generated sample paths to a YAML file.

        Converts the list of tuples to a list of lists for YAML compatibility and saves it.

        Args:
            file_path (str): Path to the YAML file.
        """
        if self.name == "split":
            sample_paths = self.split_paths()
        elif self.name == "whole":
            sample_paths = self.sample_paths
        # Convert tuples to lists for YAML compatibility
        sample_paths_as_lists = [[*t] for t in sample_paths]
        with open(file_path, 'w') as f:
            yaml.dump(sample_paths_as_lists, f)