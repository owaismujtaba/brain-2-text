import os

from src.utils import load_yaml_config, log_info
from src.logging.log import setup_logger
import pdb

if __name__ == "__main__":
    config = load_yaml_config('config.yaml')
    logger = setup_logger(
            name=config.get('logging')['name'] , 
            log_file_path=config.get('logging')['log_file_path'], 
            level=config.get('logging')['level'],
            format=config.get('logging')['format']
        )
    
    log_info(logger, "Configuration Loaded:")
    for key, value in config.items():
        if key=='run':
            logger.info(f"{key}: {value}")

    
    if config.get('run')['mode'] == 'train':
        from src.dataset.dataset import DatasetLoader
        from src.training.trainer import Trainer
        from src.training.model import BrainToTextModel
        train_loader = DatasetLoader(config, logger).get_dataloader(kind='train')
        val_loader = DatasetLoader(config, logger).get_dataloader(kind='val')

        model = BrainToTextModel(config)
        trainer = Trainer(model, config, logger)
        trainer.train(train_loader, val_loader)


    if config.get('run')['mode'] == 'inference':
        log_info(logger,'Starting Inference')
        from src.dataset.dataset import DatasetLoader
        from src.inference.inference import Inference
        pdb.set_trace()
        test_loader = DatasetLoader(config, logger).get_dataloader(kind='test')

        inferencer = Inference(
            config=config,
            logger=logger
        )

        logits, seq_class_ids, transcripts = inferencer.inference(
            dataloader=test_loader
        )



        
