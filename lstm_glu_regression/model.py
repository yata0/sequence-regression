import os
import pdb
import argparse
import numpy as np
from utils import sequence_mask_torch
import torch
import torch.nn.functional as F
from HeadDataset import HeadposeDataset
from tensorboardX import SummaryWriter
from torch import nn, optim
import Hyperparams as hp
class LstmHead(nn.Module):
    def __init__(self):
        super(LstmHead, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embed_data = nn.Embedding(hp.vocab_size, hp.embedding_dim)
        self.lstm = nn.LSTM(hp.embedding_dim,hp.hidden_dim,hp.num_layers,batch_first=True)
        self.output = nn.Linear(hp.hidden_dim,hp.target_dim)

    def forward(self,x):
        h0 = torch.zeros(hp.num_layers, x.size(0), hp.hidden_dim).to(self.device)
        c0 = torch.zeros(hp.num_layers, x.size(0), hp.hidden_dim).to(self.device)
        x = self.embed_data(x)
        out, _ = self.lstm(x, (h0,c0))
        out = F.sigmoid(self.output(out))
        return out

class LstmResHead(nn.Module):
    def __init__(self):
        super(LstmResHead, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_data = nn.Embedding(hp.vocab_size, hp.embedding_dim)
        self.lstm_1 = nn.LSTM(hp.embedding_dim, hp.hidden_dim,batch_first=True)
        self.lstm_2 = nn.LSTM(hp.hidden_dim, hp.hidden_dim,batch_first=True)
        self.output = nn.Linear(hp.hidden_dim, hp.target_dim)
    def forward(self, x):
        x = self.embed_data(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        x = self.embed_data(x)
        l1, _ = self.lstm_1(x,(h0,c0))
        l2, _ = self.lstm_2(l1,(h1,c1))
        out = F.sigmoid(self.output(l1+l2))
        return out

class Conv_LstmHead(nn.Module):
    def __init__(self,glu_layer_size,lstm_layer_size):
        super(Conv_LstmHead, self).__init__()
        self.embed_data = nn.Embedding(hp.vocab_size, hp.embedding_dim)
        self.glu_list = nn.ModuleList([])
        self.glu_list.extend([GatedLinearUnit(7, hp.embedding_dim,hp.embedding_dim*2) for i in range(glu_layer_size)])
        self.lstm_layers = nn.LSTM(hp.embedding_dim,hp.hidden_dim,lstm_layer_size,batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hp.hidden_dim,hp.target_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, length_list):
        sequences_mask = sequence_mask_torch(length_list, hp.max_len,hp.embedding_dim)
        if torch.cuda.is_available():
            sequences_mask = sequences_mask.cuda()
        # expand_dims
        out = self.embed_data(x)
        out = torch.transpose(out,1,2)
        mask = torch.transpose(sequences_mask,1,2)
        for glu in self.glu_list:
            out = glu(out)
            out = out * mask
        out = torch.transpose(out,1,2)
        lstm_out,_= self.lstm_layers(out)
        out = lstm_out + out
        out = self.output(out)
        return out

class GatedLinearUnit(nn.Module):
    """
    1d convolutional layer,with residual connect and gate mechanism
    """
    def __init__(self, kernel_size,input_dim,output_dim):
        super(GatedLinearUnit, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, 
                                out_channels=output_dim,
                                    kernel_size=kernel_size,
                                        stride = 1,
                                            padding=kernel_size//2)
    def forward(self, x):
        out = self.conv(x)
        out_a, out_b = torch.split(out, out.size(1)//2, dim=1)
        out = x + out_a * F.sigmoid(out_b)
        return out

if __name__ == "__main__":
    glu = GatedLinearUnit(7,256,256)
    print(glu)