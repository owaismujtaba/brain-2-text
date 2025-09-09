import os
import torch
import h5py
from torch.utils.data import Dataset, DataLoader

from src.dataset.utils import get_all_files


class H5pyDataset(Dataset):
    def __init__(self, config, logger, kind='train'):
        """
        Dataset for loading and padding data from HDF5 files in a directory for train/val splits.
        Expects config to be a dict-like object containing 'data_dir'.
        Instead of loading all data into memory, only indexes are stored and data is loaded at runtime.
        """
        
        self.logger = logger
        self.data_dir = config.get('dataset', {}).get('data_folder', None)
        if not self.data_dir:
            raise ValueError("Config must contain 'data_dir' key.")

        # Get all HDF5 files
        self.all_filepaths = get_all_files(self.data_dir, extensions=('.hdf5',))
        self.relevant_filepaths = self.filter_filepaths(kind)
        self.index_map = []  # List of (file_path, trial_key) tuples

        self.build_index()  # Build index of all trials

    def filter_filepaths(self, keyword):
        self.logger.info(f"Filtering files in {self.data_dir} with keyword '{keyword}'")
        filtered_files = [f for f in self.all_filepaths if keyword in os.path.basename(f)]
        if not filtered_files:
            self.logger.warning(f"No files found with keyword '{keyword}' in directory '{self.data_dir}'")
        return filtered_files

    def build_index(self):
        """Builds an index of all trials across the relevant files. Do not load all data into memory."""
        
        self.logger.info("Building index of all trials...")
        for file_path in self.relevant_filepaths:
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    self.index_map.append((file_path, key))
        self.length = len(self.index_map)
        self.logger.info(f"Indexed {self.length} trials from {len(self.relevant_filepaths)} files.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range.")
        file_path, trial_key = self.index_map[idx]
        
        with h5py.File(file_path, 'r') as f:
            g = f[trial_key]
            neural_features = torch.tensor(g['input_features'][:], dtype=torch.float32)
            n_time_steps = g.attrs.get('n_time_steps', neural_features.shape[0])
            seq_class_ids = torch.tensor(g['seq_class_ids'][:], dtype=torch.long) if 'seq_class_ids' in g else None
            seq_len = g.attrs.get('seq_len', None)
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs.get('sentence_label', None)
            session = g.attrs.get('session', None)
            block_num = g.attrs.get('block_num', None)
            trial_num = g.attrs.get('trial_num', None)

            item = {
                'neural_features': neural_features,
                'n_time_steps': n_time_steps,
                'seq_class_ids': seq_class_ids,
                'seq_len': seq_len,
                'transcriptions': transcription,
                'sentence_label': sentence_label,
                'session': session,
                'block_num': block_num,
                'trial_num': trial_num
            }
        return item


class DatasetLoader:
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config

    def get_dataloader(self, kind='train'):
        from src.dataset.utils import collate_fn
        dataset = H5pyDataset(self.config, self.logger, kind)
        batch_size = self.config.get('training', {}).get('batch_size', 16)
        dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)
        self.logger.info(f"Created DataLoader for {kind} with batch size {batch_size}")
        return dataloader