from model import LstmHead
import torch 
import torch.nn as nn
import torch.optim as optim
from utils import sequence_mask_torch,write_summary,load_model_path
from tensorboardX import SummaryWriter
from HeadDataset import HeadposeDataset
from torch.utils.data import DataLoader
import Hyperparams as hp
import numpy as np
import os


loss_function = nn.L1Loss(reduce=False) 

def train(model, dataloader, epochs,start_epoch=0):
    opti = optim.Adam(model.parameters(),lr=hp.learning_rate)
    # opti = optim.SGD(model.parameters(), lr=hp.learning_rate, momentum=0.9)
    writer = SummaryWriter(log_dir=hp.log_dir)
    dummy_data = torch.zeros([1,hp.max_len])
    

    if torch.cuda.is_available():
        model = model.cuda()
        dummy_data = dummy_data.cuda()

    # writer.add_graph(model,dummy_data)
    for epoch_index in range(start_epoch, start_epoch+epochs):
        epoch_l1_loss= train_epoch(model,dataloader,opti,epoch_index,writer)
        print("epoch:[{}/{}]\tl1 loss: {}".format(epoch_index,epochs+start_epoch,epoch_l1_loss))
        if epoch_index%10==0:
            torch.save(model.state_dict(),os.path.join(hp.save_dir,"epoch_{}_loss{}.pkl".format(epoch_index, epoch_l1_loss)))

def train_epoch(model, dataloader,optimizer, epoch_index,writer,print_every=10):

    epoch_l1_loss = 0
    print_l1_loss = 0

    batches = len(dataloader)
    # print(batches)
    for iter_index, train_data in enumerate(dataloader):
        
        iter_index = iter_index + 1
        # print(iter_index)
        total_iter_index = iter_index + epoch_index * batches
        
        optimizer.zero_grad()
        data, label_r, ori_length_list = train_data
        sequences_mask = sequence_mask_torch(ori_length_list,max_len=hp.max_len)

        data, label_r = data.long(), label_r.float()
        if torch.cuda.is_available():

            data, label_r, ori_length_list = data.cuda(),label_r.cuda(),ori_length_list.cuda()
            sequences_mask = sequences_mask.cuda()
        predictions = model(data)
        # predictions = predictions.detach()
        # print(label_r)
        # print(label_r.requires_grad)
        l1_loss = loss_function(predictions,label_r)
        # print(l1_loss.requires_grad)
        l1_loss = torch.sum(l1_loss,dim=-1)
        l1_loss_mask = l1_loss * sequences_mask
        l1_loss_final = torch.sum(l1_loss_mask)
        l1_loss = l1_loss_final/torch.sum(sequences_mask)
        

        loss = l1_loss     
        print_l1_loss += l1_loss
        
        loss.backward()
        optimizer.step()
        
        if iter_index % print_every == 0:
            print("epoch:{}\titeration:[{}\{}]\tl1_loss:{}".format(epoch_index,iter_index,len(dataloader),print_l1_loss/print_every))
            write_summary(writer,print_l1_loss/print_every,total_iter_index)
            print_l1_loss = 0

        epoch_l1_loss += l1_loss

    return epoch_l1_loss/batches

if __name__ == "__main__":
    dataset = HeadposeDataset()
    train_loader = DataLoader(dataset, batch_size=hp.batch_size,shuffle=True)
    print(len(train_loader))
    model = LstmHead()
    
    if hp.continue_train:
        model_path, start_index = load_model_path(hp.save_dir)
        model.load_state_dict(torch.load(model_path))
    else:
        start_index = 0
    
    train(model,train_loader,400,start_index)