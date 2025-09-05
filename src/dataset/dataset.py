import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class H5pyDataReader:
    def __init__(self, file_path, logger):
        import h5py
        self.logger = logger
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        self.file = h5py.File(file_path, 'r')
        self.data = self.read_all_trials()

    def get_dataset(self):
        return self.data

    def read_all_trials(self):
        self.logger.info("Reading all trials from the HDF5 file")
        data = {
            'neural_features': [],
            'n_time_steps': [],
            'seq_class_ids': [],
            'seq_len': [],
            'transcriptions': [],
            'sentence_label': [],
            'session': [],
            'block_num': [],
            'trial_num': [],
        }

        for key in self.file.keys():
            g = self.file[key]

            # Required fields
            neural_features = g['input_features'][:]
            n_time_steps = g.attrs.get('n_time_steps', neural_features.shape[0])

            # Optional fields
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs.get('seq_len', None)
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs.get('sentence_label', None)
            session = g.attrs.get('session', None)
            block_num = g.attrs.get('block_num', None)
            trial_num = g.attrs.get('trial_num', None)

            # Append to dictionary
            data['neural_features'].append(torch.tensor(neural_features, dtype=torch.float32))
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(torch.tensor(seq_class_ids, dtype=torch.long) if seq_class_ids is not None else None)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)

        return data


class H5pyDataset(Dataset):
    def __init__(self, data_dir, logger, kind='train'):
        from .utils import get_all_files  # Your utility to get all files in a dir
        self.logger = logger
        self.data_dir = data_dir

        # Get all HDF5 files
        self.all_filepaths = get_all_files(self.data_dir, extensions=('.hdf5',))
        self.relevant_filepaths = self.filter_filepaths(kind)
        self.data = None
        self.length = 0

        self.load_data()  # Load and pad data

    def filter_filepaths(self, keyword):
        self.logger.info(f"Filtering files in {self.data_dir} with keyword '{keyword}'")
        filtered_files = [f for f in self.all_filepaths if keyword in os.path.basename(f)]
        if not filtered_files:
            self.logger.warning(f"No files found with keyword '{keyword}' in directory '{self.data_dir}'")
        return filtered_files

    def load_data(self):
        all_data = {key: [] for key in [
            'neural_features', 'n_time_steps', 'seq_class_ids', 'seq_len',
            'transcriptions', 'sentence_label', 'session', 'block_num', 'trial_num']}

        # Load all files
        for file_path in self.relevant_filepaths:
            self.logger.info(f"Loading data from {file_path}")
            reader = H5pyDataReader(file_path, self.logger)
            data = reader.get_dataset()
            for key in all_data:
                all_data[key].extend(data[key])

        # Pad sequences
        all_data['neural_features'] = pad_sequence(all_data['neural_features'], batch_first=True, padding_value=0)

        # Handle optional seq_class_ids
        valid_seq_class_ids = [x for x in all_data['seq_class_ids'] if x is not None]
        if valid_seq_class_ids:
            all_data['seq_class_ids'] = pad_sequence(valid_seq_class_ids, batch_first=True, padding_value=0)
        else:
            all_data['seq_class_ids'] = None

        self.logger.info(f"features shape after padding: {all_data['neural_features'].shape}")
        self.logger.info(f"seq_class_ids shape after padding: {all_data['seq_class_ids'].shape if all_data['seq_class_ids'] is not None else 'None'}")

        self.data = all_data
        self.length = len(self.data['neural_features'])
        self.logger.info(f"Loaded {self.length} trials from {len(self.relevant_filepaths)} files.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range.")

        item = {key: self.data[key][idx] if self.data[key] is not None else None for key in self.data}
        return item


class DatasetLoader:
    def __init__(self, data_dir, logger):
        self.logger = logger
        self.data_dir = data_dir

    def get_dataloader(self, kind='train'):
        dataset = H5pyDataset(self.data_dir, self.logger, kind)
        return dataset
