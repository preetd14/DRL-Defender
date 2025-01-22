import os
import yaml

# ToDo: We can expand this class by adding more scenarios for finer-grained control on 
# the samples we use to train/evaluate the DQN agent.
# For example, we can add scenarios for split between malware and benign traces
class Scenario:
    """
    Generates and saves sample paths for different scenarios.

    Attributes:
        name (str): Name of the scenario (e.g., "train", "eval").
        root_dir (str): Root directory containing the samples.
        sample_paths (list): List to store the generated sample paths.
    """
    def __init__(self, name, root_dir):
        """
        Initializes the Scenario object.

        Args:
            name (str): Name of the scenario.
            root_dir (str): Root directory of the samples.
        """
        self.name = name
        self.root_dir = root_dir
        self.sample_paths = []

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

    def save_to_yaml(self, file_path):
        """
        Saves the generated sample paths to a YAML file.

        Converts the list of tuples to a list of lists for YAML compatibility and saves it.

        Args:
            file_path (str): Path to the YAML file.
        """
        # Convert tuples to lists for YAML compatibility
        sample_paths_as_lists = [[*t] for t in self.sample_paths]  
        with open(file_path, 'w') as f:
            yaml.dump(sample_paths_as_lists, f)  # Dump as list of lists