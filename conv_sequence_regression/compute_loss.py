import torch
import torch.nn as nn
import Hyperparams as hp
import numpy as np

loss_func_r = nn.L1Loss(reduce=False)
def gradient(out):
    gx = out[:,:-1] - out[:,1:]
    return torch.abs(gx)

def compute_loss(out_r, label_r, ori_length_lst):
    use_gpu = torch.cuda.is_available()
    curr_batch_size = out_r.size(0)
    mask_mat = torch.ones(curr_batch_size, hp.max_len)
    total_num = torch.tensor(0)
    if use_gpu:
        mask_mat  = mask_mat.cuda()
        total_num = total_num.cuda()
    for j in range(curr_batch_size):
        mask_mat[j, ori_length_lst[j]:] = 0
        total_num = total_num + ori_length_lst[j]
    
    loss_mat_r = [loss_func_r(out_r[:, :, j], label_r[:, :, j]) for j in range(3)]
    loss_r = torch.mean(mask_mat * loss_mat_r[0]) + torch.mean(mask_mat * loss_mat_r[1]) + torch.mean(mask_mat * loss_mat_r[2])
    smoothness_loss_mat = [gradient(out_r[:, :, j]) for j in range(3)]
    smooth_loss = torch.mean(mask_mat[:,:-1] * smoothness_loss_mat[0]) + torch.mean(mask_mat[:,:-1] * smoothness_loss_mat[1]) + torch.mean(mask_mat[:,:-1] * smoothness_loss_mat[2])
    loss = 0.3 * loss_r + 0.7 * smooth_loss
    return loss,loss_r, smooth_loss

if __name__ == "__main__":
    pass