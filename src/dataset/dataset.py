import os


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
        self.logger.info("Reading all trials from the HDF5 file.{self.file.filename}")
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
        keys = list(self.file.keys())
        for key in keys:
            g = self.file[key]
            neural_features = g['input_features'][:]
            n_time_steps = g.attrs['n_time_steps']
            seq_class_ids = g['seq_class_ids'][:] if 'seq_class_ids' in g else None
            seq_len = g.attrs['seq_len'] if 'seq_len' in g.attrs else None
            transcription = g['transcription'][:] if 'transcription' in g else None
            sentence_label = g.attrs['sentence_label'][:] if 'sentence_label' in g.attrs else None
            session = g.attrs['session']
            block_num = g.attrs['block_num']
            trial_num = g.attrs['trial_num']

            data['neural_features'].append(neural_features)
            data['n_time_steps'].append(n_time_steps)
            data['seq_class_ids'].append(seq_class_ids)
            data['seq_len'].append(seq_len)
            data['transcriptions'].append(transcription)
            data['sentence_label'].append(sentence_label)
            data['session'].append(session)
            data['block_num'].append(block_num)
            data['trial_num'].append(trial_num)
        return data
    

    class H5pyDataset:
        def __init__(self, dir,  logger, kind='train'):
            from .utils import get_all_files
            self.logger = logger
            self.all_filepaths = get_all_files(dir, extensions=('.hdf5'))
            self.relevanet_filepaths = self.filter_filepaths(kind)

        
        def filter_filepaths(self, keyword):
            self.logger.info(f"Filtering files in {self.dir} with keyword '{keyword}'")
            filtered_files = [f for f in self.filepaths if keyword in os.path.basename(f)]
            if not filtered_files:
                self.logger.warning(f"No files found with keyword '{keyword}' in directory '{self.dir}'")
            return filtered_files
        
        def load_data(self):
            all_data = {
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
            for file_path in self.relevanet_filepaths:
                self.logger.info(f"Loading data from {file_path}")
                reader = H5pyDataReader(file_path, self.logger)
                data = reader.get_dataset()
                for key in all_data:
                    all_data[key].extend(data[key])
            self.data = all_data
            self.length = len(self.data['neural_features'])
            self.logger.info(f"Loaded {self.length} trials from {len(self.filepaths)} files.")
            return self.data

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            if idx < 0 or idx >= self.length:
                raise IndexError("Index out of range.")
            item = {key: self.data[key][idx] for key in self.data}
            return item

