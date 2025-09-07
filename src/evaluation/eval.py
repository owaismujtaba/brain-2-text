import torch
import editdistance

def compute_phenome_error(logits, seq_class_ids, seq_lengths, phenome_seq_lengths):
    """
    Calculate the Phoneme Error Rate (PER) between true and predicted sequences.
    
    Args:
        logits (torch.Tensor): The output logits from the model of shape (T, N, C).
        seq_class_ids (torch.Tensor): The true class IDs of shape (N, S).
        seq_lengths (torch.Tensor): The lengths of the input sequences of shape (N,).
        phenome_seq_lengths (torch.Tensor): The lengths of the target phoneme sequences of shape (N,).
    """
    
    batch_edit_distance = 0
    for i in range(logits.size(0)):
        decoded = torch.argmax(logits[i, :seq_lengths[i], :], dim=-1)
        decoded = torch.unique_consecutive(decoded, dim=-1)
        decoded = decoded.cpu().numpy()
        decoded = decoded[decoded != 0]  # remove blanks

        true_seq = seq_class_ids[i, :phenome_seq_lengths[i]].cpu().numpy()

        batch_edit_distance += editdistance.eval(decoded, true_seq)
    
    total_phenomes = phenome_seq_lengths.sum().item()
    per = batch_edit_distance / total_phenomes if total_phenomes > 0 else 0.0

    return per