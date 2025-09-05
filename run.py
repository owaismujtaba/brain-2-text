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
        from src.training.trainer import Trainer
        from src.training.model import BrainToTextModel
        train_loader = DatasetLoader(config, logger).get_dataloader(kind='train')
        val_loader = DatasetLoader(config, logger).get_dataloader(kind='val')

        model = BrainToTextModel(config)
        trainer = Trainer(model, config)
        trainer.train(train_loader, val_loader)

        