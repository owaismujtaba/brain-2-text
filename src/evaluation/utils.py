import torch
import torchaudio.functional as F


LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    ' | ',
]



def compute_phenome_error(logits, seq_class_ids, seq_lengths, phenome_seq_lengths):
    """
    Calculate the Phoneme Error Rate (PER) between true and predicted sequences.
    
    Args:
        logits (torch.Tensor): Model output logits of shape (B, T, C).
                               B = batch size, T = time steps, C = classes.
        seq_class_ids (torch.Tensor): Ground-truth class IDs of shape (B, S).
        seq_lengths (torch.Tensor): Valid lengths of each input sequence (B,).
        phenome_seq_lengths (torch.Tensor): Valid lengths of each target phoneme sequence (B,).
    
    Returns:
        float: Average PER across the batch.
    """
    
    batch_edit_distance = 0
    total_phonemes = int(phenome_seq_lengths.sum().item())

    for i in range(logits.size(0)):
        decoded = torch.argmax(logits[i, :seq_lengths[i]], dim=-1)
        decoded = torch.unique_consecutive(decoded)
        decoded = decoded[decoded != 0]
        true_seq = seq_class_ids[i, :phenome_seq_lengths[i]]
        batch_edit_distance += F.edit_distance(decoded, true_seq)

    
    per = batch_edit_distance / total_phonemes if total_phonemes > 0 else 0.0
    return per