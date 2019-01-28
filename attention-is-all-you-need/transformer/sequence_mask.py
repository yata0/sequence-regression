import torch
import numpy as np

def sequence_mask_torch(length_list, max_len = None, n_dims=None,dtype=np.float32):
    """

    """

    if isinstance(length_list, torch.Tensor):
        
        if torch.cuda.is_available():
            lengths = length_list.cpu().numpy()
        else:
            lengths = length_list.numpy()
    else:
        lengths = np.asarray(length_list)

    if max_len is None:
        max_len = np.max(lengths)

    row_vector = np.arange(0, max_len)
    lengths = np.squeeze(lengths)
    lengths = np.expand_dims(lengths, axis=-1)

    mask_matrix = row_vector < lengths
    mask_matrix = mask_matrix.astype(dtype)

    if n_dims != None:
        mask_matrix = np.expand_dims(mask_matrix, -1)
        mask_matrix = np.tile(result_matrix, (1, 1, n_dims))
    return torch.from_numpy(mask_matrix)

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

if __name__ == "__main__":

    a = [1,2,3,4,5]
    print(sequence_mask_torch(a))
    