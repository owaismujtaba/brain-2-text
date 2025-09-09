import os
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperProcessor
import numpy as np
import pdb

def collate_fn(batch):
    """
    Collates a list of dicts from H5pyDataset into a batch.
    Pads 'neural_features' along the time axis (dim=0).
    """
    # Extract features and their lengths
    features = [item['neural_features'] for item in batch]  # list of [T_i, C]
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    sentence_labels = [item['sentence_label'] for item in batch]

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    #sentence_labels = processor.tokenizer(sentence_labels, return_tensors="pt", padding=True).input_ids
    sentence_labels, sentence_len = get_sentence_ids(processor, sentence_labels)

    # Pad sequences -> [B, T_max, C]
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)  
    sentrence_labels = pad_sequence(sentence_labels, batch_first=True, padding_value=0)
    
    batch_out = {
        'neural_features': features_padded,
        'seq_lengths': lengths,
        'sentence_label': sentrence_labels,
        'sentence_len': sentence_len
    }

    '''
    # Collate all other keys
    for key in batch[0].keys():
        if key == 'neural_features' or key == 'sentence_label' or key == 'seq_lengths':
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
    '''

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

def get_sentence_ids(processor, sentences):
    sentence_labels = []
    sentence_len = []
    for sentence in sentences:
        labels = processor.tokenizer(sentence, return_tensors="pt").input_ids
        labels = labels.squeeze()
        sentence_len.append(labels.size(0))
        sentence_labels.append(labels)

    return sentence_labels, sentence_len