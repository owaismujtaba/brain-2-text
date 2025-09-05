import os

from utils import load_yaml_config
from src.logging.log import setup_logger
import pdb
import torch

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
        from src.dataset.dataset import DatasetLoader
        
        train_loader = DatasetLoader(config, logger).get_dataloader(kind='train')
        val_loader = DatasetLoader(config, logger).get_dataloader(kind='val')

        for batch in train_loader:
            print("Batch from training loader:")
            for key, value in batch.items():
                if isinstance(value, list):
                    print(f"{key}: List of length {len(value)},{value[:2] if len(value)<=2 else '...'}")
                elif isinstance(value, torch.Tensor):
                    print(f"{key}: Tensor of shape {value.shape}")
                else:
                    print(f"{key}: {value}")
            import pdb; pdb.set_trace()
            