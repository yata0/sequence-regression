# -*- coding: utf-8 -*-

import os
import pdb
import random
import argparse

import numpy as np
import pandas as pd

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from HeadDataSet import HeadposeDataset
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import Hyperparams as hp

class ConvRegression(nn.Module):
    def __init__(self):
        super(ConvRegression,self).__init__()
        self.use_gpu = torch.cuda.is_available()
        self.embed_data = nn.Embedding(hp.vocab_size,hp.embedding_dim)
        self.embed_position = nn.Embedding.from_pretrained(self.embed_position_table(hp.max_len+1, padding_index=0))
        self.conv1 = nn.Conv1d(in_channels=hp.embedding_dim,out_channels=hp.embedding_dim*2,kernel_size=7,stride=1,padding=3)
        self.conv2 = nn.Conv1d(in_channels=hp.embedding_dim,out_channels=hp.embedding_dim*2,kernel_size=7,stride=1,padding=3)
        self.conv3 = nn.Conv1d(in_channels=hp.embedding_dim,out_channels=hp.embedding_dim*2,kernel_size=7,stride=1,padding=3)
        self.conv4 = nn.Conv1d(in_channels=hp.embedding_dim,out_channels=hp.embedding_dim*2,kernel_size=7,stride=1,padding=3)
        self.regress = nn.Sequential(
            nn.Linear(hp.hidden_dim,hp.target_dim),
            nn.Sigmoid()
        )
    def forward(self, data):
        
        if hp.pe:

            out = self.embed_data(data) + self.embed_position(position)

        else:
            out = self.embed_data(data)
        # [batch_size,max_len,embed_dims]----[batch_size,embed_dims,max_len]
        
        out = torch.transpose(out,1,2)
        conv1 = self.conv1(out)
        out_a,out_b= torch.split(conv1, hp.embedding_dim, dim=1)
        out = out + out_a * torch.sigmoid(out_b)

        conv2 = self.conv2(out)
        out_a, out_b = torch.split(conv2,hp.embedding_dim,dim=1)
        out = out + out_a * torch.sigmoid(out_b)

        conv3 = self.conv3(out)
        out_a, out_b = torch.split(conv3, hp.embedding_dim, dim=1)
        out = out + out_a * torch.sigmoid(out_b)

        conv4 = self.conv4(out)
        out_a, out_b = torch.split(conv4, hp.embedding_dim,dim=1)
        out = out + out_a * torch.sigmoid(out_b)
        
        out = torch.transpose(out, 1,2)
        prediction = self.regress(out)
        return prediction   

    def embed_position_table(self,position_len,padding_index=None):

        # pe_table = [()]
        s_tables = np.zeros([position_len, hp.embedding_dim])
        def get_position(position_index):
            pe_table = [position_index/np.power(float(10000),2*(i//2)/hp.embedding_dim) for i in range(hp.embedding_dim)]
            return pe_table
        tables = np.array([get_position(pos_index) for pos_index in range(position_len)])    
        s_tables[:, 0::2] = np.sin(tables[:,0::2])
        s_tables[:, 1::2] = np.cos(tables[:,1::2])
        if padding_index is not None:
            s_tables[padding_index] = 0
        return torch.FloatTensor(s_tables) 