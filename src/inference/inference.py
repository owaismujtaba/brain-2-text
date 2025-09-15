import torch
from pathlib import Path
from tqdm import tqdm
from src.training.model import BrainToTextModel


class Inference:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model = BrainToTextModel(config=config)
        self._device_setup()
        self.model.to(self.device)
        self._load_model_checkpoint()



    def _device_setup(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _load_model_checkpoint(self):
        self.logger.info("Loading best_model checkpoint")
        checkpoint_dir = self.config.get('training', {}).get('checkpoints_dir')
        checkpoint_path = Path(checkpoint_dir, 'best_model.pt')
        self.logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path} or set load_checkpoint False"
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
                
        self.logger.info(f"Model loaded sucessfully")

    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation"""

        inputs = batch['neural_features']
        seq_class_ids = batch['seq_class_ids']
        transcripts = batch['sentence_label']
        inputs = inputs.to(self.device)
        seq_class_ids = seq_class_ids.to(self.device)
        transcripts = transcripts.to(self.device)

        return inputs, seq_class_ids, transcripts 
    

    
    def inference(self, dataloader):
        self.logger.info("Starting inference")
        

        self.logits_full = []
        self.seq_class_ids_full = []
        self.transcripts_full = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Inference", unit="batch")
            for batch in pbar:
                inputs, seq_class_ids, transcripts = self._prepare_batch(batch)

                logits = self.model(inputs)
                self.logits_full.append(logits)
                self.seq_class_ids_full.append(seq_class_ids)
                self.transcripts_full.append(transcripts)

        self.logger.info("Inferencde completed")
