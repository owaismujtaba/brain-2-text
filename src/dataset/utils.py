import os
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Collates a list of dicts from H5pyDataset into a batch.
    Pads 'neural_features' along the time axis (dim=0).
    """
    # Extract features and their lengths
    features = [item['neural_features'] for item in batch]  # list of [T_i, C]
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)

    # Pad sequences -> [B, T_max, C]
    features_padded = pad_sequence(features, batch_first=True)  

    batch_out = {
        'neural_features': features_padded,
        'seq_lengths': lengths,
    }

    # Collate all other keys
    for key in batch[0].keys():
        if key == 'neural_features':
            continue
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            try:
                batch_out[key] = torch.stack(values)
            except RuntimeError:
                # e.g., if variable-length labels like seq_class_ids
                batch_out[key] = values
        else:
            batch_out[key] = values

    return batch_out


def get_all_files(parent_dir, extensions=None):
    """
    Recursively loads all files in the parent directory and its subdirectories.
    If extensions is provided, only files with those extensions are returned.

    Args:
        parent_dir (str): Path to the parent directory.
        extensions (tuple, optional): File extensions to filter by, e.g. ('.hdf5')

    Returns:
        List[str]: List of file paths.
    """
    all_files = []
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if extensions is None or file.lower().endswith(extensions):
                all_files.append(os.path.join(root, file))
                
    return all_files