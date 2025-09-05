import os

from utils import load_yaml_config
from src.logging.log import setup_logger
import pdb


if __name__ == "__main__":
    config = load_yaml_config('config.yaml')
    print("Configuration Loaded:")
    for key, value in config.items():
        print(f"{key}: {value}")

    logger = setup_logger(
            name=config.get('logging')['name'] , 
            log_file_path=config.get('logging')['log_file_path'], 
            level=config.get('logging')['level'],
            format=config.get('logging')['format']
        )
    if config.get('run')['mode'] == 'train':
        print("Starting training process...")
        from src.dataset.dataset import H5pyDataset
        from src.logging.log import setup_logger
        

        dataset = H5pyDataset(
            data_dir=config.get('dataset')['data_folder'], 
            logger=logger, kind='train'
        )