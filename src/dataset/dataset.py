import os
import math
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.dataset.utils import get_all_files
import pdb

class H5pyBatchDataset(Dataset):
    """
    Dataset that loads batches of trials from multiple HDF5 files.
    """

    def __init__(
        self, config, logger, kind="train"
    ):
        """
        Args:
            config (dict): Configuration dictionary.
            logger: Logger object for logging.
            kind (str): "train" or "test" split.
        """
        self.config = config
        self.logger = logger
        self.kind = kind
        self.logger.info("Initializing H5pyBatchDataset")
        pdb.set_trace()
        self._setup_configuration_parameters()
        # build index of all trials
        self.index_map, self.trial_indices = self.build_index()
        self.n_days = len(self.trial_indices)
        
        

        if self.kind == "train" and self.must_include_days > self.n_days:
            raise ValueError(f"days_per_batch={self.must_include_days} > n_days={self.n_days}")

        # --- Build batch index ---
        if self.split == "train":
            self.batch_index = self._create_batch_index_train()
        else:
            self.batch_index = self._create_batch_index_test()
            self.n_batches = len(self.batch_index)

    def _setup_configuration_parameters(self):
        self.data_dir = self.config.get('dataset', {}).get('data_folder', None)
        self.must_include_days = self.config.get('training', {}).get('must_include_days', 5)
        if not self.data_dir:
            raise ValueError("Config must contain 'data_dir' key.")

        self.all_filepaths = get_all_files(self.data_dir, extensions=('.hdf5',))
        self.relevant_filepaths = self.filter_filepaths(self.kind)
        self.n_days = len(self.relevant_filepaths)

    def filter_filepaths(self, keyword):
        self.logger.info(f"Filtering files in {self.data_dir} with keyword '{keyword}'")
        filtered_files = [f for f in self.all_filepaths if keyword in os.path.basename(f)]
        if not filtered_files:
            raise ValueError(f"No files found in {self.data_dir} with keyword '{keyword}'")
        self.logger.info(f"Found {len(filtered_files)} files for '{keyword}' split.")
        return filtered_files
    # -----------------------------------------------------------------
    # Index building
    # -----------------------------------------------------------------
    def build_index(self):
        """
        Build trial index for each file.
        Returns:
            index_map: [(file_path, trial_key, day_id), ...]
            trial_indices: {day: {"trials": [trial_keys], "session_path": str}}
        """
        index_map = []
        trial_indices = {}

        day_id = 0
        for file_path in self.filepaths:
            with h5py.File(file_path, "r") as f:
                trial_keys = list(f.keys())
                for k in trial_keys:
                    index_map.append((file_path, k, day_id))
                trial_indices[day_id] = {"trials": trial_keys, "session_path": file_path}
            day_id += 1

        if self.logger:
            self.logger.info(f"Indexed {len(index_map)} trials from {len(filepaths)} days.")

        return index_map, trial_indices

    # -----------------------------------------------------------------
    # Dataset API
    # -----------------------------------------------------------------
    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        """Return one full batch (dict)."""
        batch = {
            "input_features": [],
            "seq_class_ids": [],
            "n_time_steps": [],
            "phone_seq_lens": [],
            "day_indices": [],
            "transcriptions": [],
            "block_nums": [],
            "trial_nums": [],
        }

        index = self.batch_index[idx]

        for d, trial_list in index.items():
            with h5py.File(self.trial_indices[d]["session_path"], "r") as f:
                for t in trial_list:
                    g = f[t]

                    feats = torch.from_numpy(g["input_features"][:])
                    if self.feature_subset is not None:
                        feats = feats[:, self.feature_subset]
                    batch["input_features"].append(feats)

                    if "seq_class_ids" in g:
                        batch["seq_class_ids"].append(torch.from_numpy(g["seq_class_ids"][:]))
                    else:
                        batch["seq_class_ids"].append(torch.empty(0, dtype=torch.long))

                    if "transcription" in g:
                        batch["transcriptions"].append(torch.from_numpy(g["transcription"][:]))
                    else:
                        batch["transcriptions"].append(torch.empty(0, dtype=torch.long))

                    batch["n_time_steps"].append(g.attrs.get("n_time_steps", feats.shape[0]))
                    batch["phone_seq_lens"].append(g.attrs.get("seq_len", 0))
                    batch["day_indices"].append(int(d))
                    batch["block_nums"].append(g.attrs.get("block_num", -1))
                    batch["trial_nums"].append(g.attrs.get("trial_num", -1))

        # --- Pad to tensors ---
        batch["input_features"] = pad_sequence(batch["input_features"], batch_first=True, padding_value=0)
        batch["seq_class_ids"] = pad_sequence(batch["seq_class_ids"], batch_first=True, padding_value=0)
        batch["transcriptions"] = pad_sequence(batch["transcriptions"], batch_first=True, padding_value=0)

        batch["n_time_steps"] = torch.tensor(batch["n_time_steps"])
        batch["phone_seq_lens"] = torch.tensor(batch["phone_seq_lens"])
        batch["day_indices"] = torch.tensor(batch["day_indices"])
        batch["block_nums"] = torch.tensor(batch["block_nums"])
        batch["trial_nums"] = torch.tensor(batch["trial_nums"])

        return batch

    # -----------------------------------------------------------------
    # Batch index creation
    # -----------------------------------------------------------------
    def _create_batch_index_train(self):
        """Random batch assignment across days."""
        batch_index = {}
        non_must_days = (
            [d for d in self.trial_indices if d not in self.must_include_days]
            if self.must_include_days else list(self.trial_indices.keys())
        )

        for b in range(self.n_batches):
            # --- choose days ---
            if self.must_include_days:
                extra_days = np.random.choice(
                    non_must_days,
                    size=self.days_per_batch - len(self.must_include_days),
                    replace=False,
                )
                days = np.concatenate([self.must_include_days, extra_days])
            else:
                days = np.random.choice(
                    list(self.trial_indices.keys()),
                    size=self.days_per_batch,
                    replace=False,
                )

            # --- allocate trials ---
            per_day = math.ceil(self.batch_size / self.days_per_batch)
            batch = {}
            for d in days:
                trial_keys = np.random.choice(
                    self.trial_indices[d]["trials"],
                    size=per_day,
                    replace=True,
                )
                batch[d] = trial_keys

            # trim to exact batch_size
            while sum(len(v) for v in batch.values()) > self.batch_size:
                d = np.random.choice(days)
                if len(batch[d]) > 1:
                    batch[d] = batch[d][:-1]

            batch_index[b] = batch
        return batch_index

    def _create_batch_index_test(self):
        """Sequential batching per day (no mixing)."""
        batch_index = {}
        b = 0
        for d in self.trial_indices:
            trials = self.trial_indices[d]["trials"]
            for start in range(0, len(trials), self.batch_size):
                batch_index[b] = {d: trials[start:start + self.batch_size]}
                b += 1
        return batch_index
