import numpy as np
import torch
def sequence_mask(lengths,max_len=None,type_=np.bool):
    lengths = np.asarray(lengths)
    if max_len is None:
        max_len = np.max(lengths)
    row_vector = np.arange(0,max_len)
    result = np.expand_dims(lengths,axis=-1)
    result_matrix = row_vector < result
    result_matrix = result_matrix.astype(type_)
    return result_matrix

def sequence_mask_torch(lengths,max_len=None,type_=np.float32):
    # lengths = np.asarray(lengths)
    lengths= lengths.numpy()
    lengths = np.asarray(lengths)
    if max_len is None:
        max_len = np.max(lengths)
    row_vector = np.arange(0,max_len)
    result = np.expand_dims(lengths,axis=-1)
    result_matrix = row_vector < result
    result_matrix = result_matrix.astype(type_)
    return torch.from_numpy(result_matrix)

if __name__ == "__main__":
    print(sequence_mask_torch(torch.tensor([1,2,2,3,3,3,4,3]),max_len=6,type_=np.float32))
