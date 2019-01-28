from Model import ConvRegression
import torch 
import torch.nn as nn
import torch.optim as optim
from utils import sequence_mask_torch,write_summary,load_model_path
from tensorboardX import SummaryWriter
from HeadDataSet import HeadposeDataset
from torch.utils.data import DataLoader
import Hyperparams as hp
import numpy as np
import os
from compute_loss import compute_loss
loss_function = nn.L1Loss(reduce=False) 
def get_smooth_loss(output):
    gx = output[:,:-1,:] - output[:,1:,:]
    return torch.abs(gx)
def train(model, dataloader, epochs,start_epoch=0):
    opti = optim.Adam(model.parameters(),lr=hp.learning_rate)
    # opti = optim.SGD(model.parameters(), lr=hp.learning_rate, momentum=0.9)
    writer = SummaryWriter(log_dir=hp.log_dir)
    dummy_data = torch.zeros([1,hp.max_len])

    dummy_data = dummy_data.long()
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_data = dummy_data.cuda()

    writer.add_graph(model,dummy_data)

    for epoch_index in range(start_epoch, start_epoch+epochs):
        epoch_l1_loss,epoch_smooth_loss,epoch_loss = train_epoch(model,dataloader,opti,epoch_index,writer)
        print("epoch:[{}/{}]\tl1 loss: {}\tsmooth_loss: {}\ttotal_loss: {}".format(epoch_index,epochs+start_epoch,epoch_l1_loss,epoch_smooth_loss,epoch_loss))
        if epoch_index % 10 == 0:
            torch.save(model.state_dict(),os.path.join(hp.save_dir,"epoch_{}_loss{}.pkl".format(epoch_index, epoch_l1_loss)))

def train_epoch(model, dataloader,optimizer, epoch_index,writer,print_every=10):
    epoch_loss = 0
    epoch_l1_loss = 0
    epoch_smooth_loss = 0

    print_loss = 0
    print_l1_loss = 0
    print_smooth_loss = 0
    batches = len(dataloader)

    for iter_index, train_data in enumerate(dataloader):
        iter_index += 1
        total_iter_index = iter_index + epoch_index * batches
        
        optimizer.zero_grad()

        data, position,label_r,ori_length_list= train_data
        sequences_mask = sequence_mask_torch(ori_length_list,max_len=hp.max_len)
        # print(sequences_mask)
        # print("sequence_mask.shape:")
        # print(sequences_mask.shape)
        data,position,label_r = data.long(), position.long(),label_r.float()
        if torch.cuda.is_available():

            data, position,label_r,ori_length_list = data.cuda(),position.cuda(), label_r.cuda(),ori_length_list.cuda()
            sequences_mask = sequences_mask.cuda()
        # print("sequence mask size:")
        # print(sequences_mask.size())
        predictions = model(data)
        loss,l1_loss,smooth_loss = compute_loss(predictions, label_r, ori_length_list)        
        # l1_loss = loss_function(label_r,predictions)
        # print("l1 loss size:")
        # print(l1_loss.size())
        # l1_loss = torch.sum(l1_loss,dim=-1)
        # l1_loss_mask = l1_loss * sequences_mask
        # l1_loss_final = torch.sum(l1_loss_mask)
        # l1_loss = l1_loss_final/torch.sum(sequences_mask)
        
        # smooth_loss = get_smooth_loss(predictions)
        # smooth_loss = torch.sum(smooth_loss,dim=-1)
        # smooth_loss = torch.sum(smooth_loss*sequences_mask[:,1:])/torch.sum(sequences_mask[:,1:])
        # loss = 0.3 * l1_loss + 0.7*smooth_loss
       
        print_l1_loss += l1_loss
        print_loss += loss
        print_smooth_loss +=smooth_loss
        
        loss.backward()
        optimizer.step()
        
        if iter_index % print_every == 0:
            print("epoch:{}\titeration:[{}\{}]\tl1_loss:{}\tsmooth_loss:{}\tloss:{}".format(
                epoch_index,iter_index,len(dataloader),print_l1_loss/print_every,print_smooth_loss/print_every,print_loss/print_every
            ))
            write_summary(writer,print_l1_loss/print_every,print_smooth_loss/print_every,print_loss/print_every,total_iter_index)
            print_l1_loss = 0
            print_loss = 0
            print_smooth_loss=0

        epoch_loss += loss
        epoch_l1_loss += l1_loss
        epoch_smooth_loss += smooth_loss

    return epoch_l1_loss/batches, epoch_smooth_loss/batches,epoch_loss/batches

if __name__ == "__main__":
    dataset = HeadposeDataset()
    train_loader = DataLoader(dataset, batch_size=hp.batch_size,shuffle=True)
    model = ConvRegression()
    if hp.continue_train:
        model_path, start_index = load_model_path(hp.save_dir)
        model.load_state_dict(torch.load(model_path))
    else:
        start_index = 0
    train(model,train_loader,200,start_index)