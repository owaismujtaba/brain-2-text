import os

from utils import load_yaml_config
from src.logging.log import setup_logger
import pdb
import torch

if __name__ == "__main__":
    data_dir = '/kaggle/input/brain-to-text-25/t15_copyTask_neuralData/hdf5_data_final'
    
    config = load_yaml_config('/kaggle/working/brain-2-text/config.yaml')
    config['dataset']['data_dir'] = data_dir
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
        logger.info("*************Mode: Train*************")
        from src.dataset.dataset import DatasetLoader
        from src.training.trainer import Trainer
        from src.training.model import BrainToTextModel
        train_loader = DatasetLoader(config, logger).get_dataloader(kind='train')
        val_loader = DatasetLoader(config, logger).get_dataloader(kind='val')
        
        model = BrainToTextModel(config)
        trainer = Trainer(model, config, logger)
        trainer.train(train_loader, val_loader)

    if config.get('run')['mode'] == 'test':
        pass
        