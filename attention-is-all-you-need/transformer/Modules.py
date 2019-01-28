import torch
import torch.nn as nn
from transformer.SubModules import MultiHeadAttention
from transformer.SubModules import FeedForwardNetworks
import numpy as np
def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multiheadAttn = MultiHeadAttention(n_head, dim_model, d_k, d_v)
        self.position_ffn = FeedForwardNetworks(dim_model, dim_inner)

    def forward(self, enc_input, zero_pad_mask, n_inf_pad_mask):
        encoder_output, att_matrix = self.multiheadAttn(
            enc_input, enc_input, enc_input, n_inf_pad_mask
        )
        enc_output = encoder_output * zero_pad_mask
        
        enc_output = self.position_ffn(enc_output)
        enc_output *= zero_pad_mask

        return enc_output, att_matrix


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.embedding_layer = nn.Embedding(opt.vocab_size, opt.embedding_size, padding_idx = 0)
        n_position = opt.max_len + 1

        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, opt.embedding_size, padding_idx=0),
            freeze=True
        )
        
        self.MultiLayerEncoder = nn.ModuleList([
            EncoderLayer(opt.dim_model, opt.dim_inner, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout)
            for _ in range(opt.num_layers) 
        ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(opt.dim_model, opt.target_dim),
            nn.Tanh()
        )
    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.embedding_layer(src_seq) + self.position_embedding(src_pos)

        for enc_layer in self.MultiLayerEncoder:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask,
                slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        enc_output = self.output_layer(enc_output)
        
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output
