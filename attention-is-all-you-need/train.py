from transformer.Modules import Encoder
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

def smmoth_loss(predictions, sequence_mask):

    loss = (predictions[:,:-1,:] - predictions[:,1:,:]) ** 2
    loss = torch.sum(loss, dim=-1)
    loss *= sequence_mask
    loss = torch.sum(loss) / torch.sum(sequence_mask)
    return loss
    
    


def train(model, dataloader, epochs,start_epoch=0):
    opti = optim.Adam( filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09)
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
        if epoch_index % 10==0:
            torch.save(model.state_dict(),os.path.join(hp.save_dir,"epoch_{}_loss{}.pkl".format(epoch_index, epoch_l1_loss)))

def train_epoch(model, dataloader,optimizer, epoch_index,writer,print_every=10):

    epoch_l1_loss = 0
    print_l1_loss = 0

    batches = len(dataloader)
    # print(batches)
    model.train()
    for iter_index, train_data in enumerate(dataloader):
        
        iter_index = iter_index + 1
        # print(iter_index)
        total_iter_index = iter_index + epoch_index * batches
        
        optimizer.zero_grad()
        data, label_r, position_idx, ori_length_list = train_data
        sequences_mask = sequence_mask_torch(ori_length_list,max_len=hp.max_len)

        data, label_r, position_idx = data.long(), label_r.float(),position_idx.long()
        if torch.cuda.is_available():

            data, label_r, position_idx, ori_length_list = data.cuda(),label_r.cuda(),position_idx.cuda(),ori_length_list.cuda()
            sequences_mask = sequences_mask.cuda()
        predictions = model(data, position_idx)
        # predictions = predictions.detach()
        # print(label_r)
        # print(label_r.requires_grad)
        l1_loss = loss_function(predictions,label_r)
        smooth_loss_ = smmoth_loss(predictions, sequences_mask[:, 1:])
        # print(l1_loss.requires_grad)
        l1_loss = torch.sum(l1_loss,dim=-1)
        l1_loss_mask = l1_loss * sequences_mask
        l1_loss_final = torch.sum(l1_loss_mask)
        l1_loss = l1_loss_final/torch.sum(sequences_mask)
        

        loss = l1_loss * 0.95  + smooth_loss_ * 0.05
        # 防止loss直接累计泄露内存

        print_l1_loss += float(l1_loss)
        epoch_l1_loss += float(l1_loss)

        loss.backward()
        optimizer.step()
        
        if iter_index % print_every == 0:
            print("epoch:{}\titeration:[{}\{}]\tl1_loss:{}\tsmooth_loss:{}".format(epoch_index,iter_index,len(dataloader),print_l1_loss/print_every,smooth_loss_))
            write_summary(writer,print_l1_loss/print_every,total_iter_index)
            print_l1_loss = 0
        # 防止loss直接累计泄露内存
        # epoch_l1_loss += float(l1_loss)

    return epoch_l1_loss/batches

if __name__ == "__main__":
    dataset = HeadposeDataset()
    train_loader = DataLoader(dataset, batch_size=hp.batch_size,shuffle=True)
    print(len(train_loader))
    model = Encoder(hp)
    
    if hp.continue_train:
        model_path, start_index = load_model_path(hp.save_dir)
        model.load_state_dict(torch.load(model_path))
    else:
        start_index = 0
    print(model)
    train(model,train_loader,201 ,start_index)
