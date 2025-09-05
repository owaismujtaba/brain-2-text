
import os

def load_yaml_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    import yaml

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"The configuration file {config_path} does not exist.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config

