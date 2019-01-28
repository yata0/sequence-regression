import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):

    """
    根据Q,K,V计算attention
    """
    def __init__(self, scale, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask):
        
        n_dim = q.size(-1)
        score = torch.bmm(q, k.transpose(1,2))
        
        if self.scale:
            score = score / np.sqrt(n_dim)
        
        if mask is not None:
            score = score.masked_fill(mask, -np.inf)
        
        attn = self.softmax(score)
        attn = self.dropout(attn)
        out = torch.bmm(attn, v)

        return out, attn

class MultiHeadAttention(nn.Module):
    """
    multi head attention 
    输入的维度均为dim_model
    q和k的维度必须映射为一致，所以两个共享一个d_k即可

    """
    def __init__(self,n_head, dim_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.n_head = n_head
        self.dim_model = dim_model
        self.d_k = d_k
        self.d_v = d_v

        self.Wq = nn.Linear(self.dim_model, self.n_head * self.d_k)
        self.Wk = nn.Linear(self.dim_model, self.n_head * self.d_k)
        self.Wv = nn.Linear(self.dim_model, self.n_head * self.d_v)

        nn.init.normal_(self.Wq.weight, mean=0, std=np.sqrt(2.0 / (dim_model + d_k)))
        nn.init.normal_(self.Wk.weight, mean=0, std=np.sqrt(2.0 / (dim_model + d_k)))
        nn.init.normal_(self.Wv.weight, mean=0, std=np.sqrt(2.0 / (dim_model + d_v)))

        self.attention = ScaledDotProductAttention(scale=True)

        self.layer_norm = nn.LayerNorm(self.dim_model)

        self.fc = nn.Linear(n_head*d_v, dim_model)

        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):

        batch_size, len_q,_ = Q.size()
        len_k = K.size(1)
        len_v = V.size(1)

        residual_input = Q

        Q = self.Wq(Q).view(batch_size, len_q, self.n_head, self.d_k)
        K = self.Wk(K).view(batch_size, len_k, self.n_head, self.d_k)
        V = self.Wv(V).view(batch_size, len_v, self.n_head, self.d_v)
        # 有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
        Q = Q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k)
        K = K.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k)
        V = V.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v)

        mask = mask.repeat(self.n_head, 1 ,1)
        output, attn = self.attention(Q, K, V, mask = mask)

        output = output.view(self.n_head, batch_size, len_q, self.d_v)

        output = output.permute(1,2,0,3).contiguous().view(batch_size, len_q, -1)

        output = self.dropout(self.fc(output))

        output = self.layer_norm(output + residual_input)
        
        return output, attn

class FeedForwardNetworks(nn.Module):
    def __init__(self, dim_model, hidden_dim, dropout=0.1):
        super(FeedForwardNetworks, self).__init__()
        self.conv1 = nn.Conv1d(dim_model, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, dim_model, 1)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual_input = x
        output = x.transpose(1, 2)
        output = self.conv2(F.relu(self.conv1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual_input)
        return output


        




        


