import os
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperProcessor
import numpy as np
import pdb

MODEL_NAME = "openai/whisper-base"  
LANG = "en"
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANG, task="transcribe")


def collate_fn(batch):
    """
    Collates a list of dicts from H5pyDataset into a batch.
    Pads 'neural_features' along the time axis (dim=0).
    """
    # Extract features and their lengths
    features = [item['neural_features'] for item in batch]  # list of [T_i, C]
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    sentence_labels = [item['sentence_label'] for item in batch]

    # Tokenize sentence labels as Whisper decoder targets
    labels = processor.tokenizer(
        text_target=sentence_labels,   # <-- use text_target explicitly
        add_special_tokens=True,       # adds <|startoftranscript|>, language, task, etc.
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).input_ids

    # Pad neural features
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)

    batch_out = {
        'neural_features': features_padded,
        'seq_lengths': lengths,
        'tok_labels': labels,             # <-- return labels directly
        'sentence_label': sentence_labels,  # keep raw text if you still need it
    }

    # Collate other keys if present
    for key in batch[0].keys():
        if key in ['neural_features', 'sentence_label', 'seq_lengths']:
            continue
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            try:
                batch_out[key] = torch.stack(values)
            except RuntimeError:
                batch_out[key] = values  # variable-length
        else:
            batch_out[key] = values

    return batch_out

def pad_to_mel_length(mel, mel_frames=3000):
    B, n_mels, T = mel.shape
    if T < mel_frames:
        pad_amount = mel_frames - T
        mel = torch.nn.functional.pad(mel, (0, pad_amount))  # pad on last dim
    else:
        mel = mel[:, :, :mel_frames]  # truncate if too long
    return mel


def get_all_files(parent_dir, kind, extensions=None,):
    """
    Recursively loads all files in the parent directory and its subdirectories.
    If extensions is provided, only files with those extensions are returned.

    Args:
        parent_dir (str): Path to the parent directory.
        extensions (tuple, optional): File extensions to filter by, e.g. ('.hdf5')

    Returns:
        List[str]: List of file paths.
    """
    filepaths = []
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if extensions is None or file.lower().endswith(extensions):
                if kind in file:
                    filepaths.append(os.path.join(root, file))
                
    return filepaths

def get_sentence_ids(processor, sentences):
    pdb.set_trace()
    sentence_labels = []
    sentence_len = []
    for sentence in sentences:
        labels = processor.tokenizer(sentence, return_tensors="pt").input_ids
        labels = labels.squeeze()
        sentence_len.append(labels.size(0))
        sentence_labels.append(labels)

    return sentence_labels, sentence_len