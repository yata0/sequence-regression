from Model import ConvRegression
from utils import input_phoneme_file,process_head_result,process_for_new,process_for_new_siyuanshu
import torch
import os
import glob

import numpy as np
def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)

def inference(test_file_list,model_path,target_root):
    mkdir(target_root)
    file_list = []
    file_list.extend(test_file_list)
    model = ConvRegression()
    model.load_state_dict(torch.load(model_path))
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use cuda")
        model = model.cuda()
    model.eval()
    for file_ in file_list:
        data = input_phoneme_file(file_,True)
        data = np.array(data).astype("int")
        position = np.arange(data.shape[0]) + 1
        data = torch.from_numpy(data)
        position = torch.from_numpy(position)
        data = torch.unsqueeze(data, 0)
        position = torch.unsqueeze(position, 0)
        data,position = data.long(),position.long()
        if use_gpu:
            data = data.cuda()
            position = position.cuda()
        prediction = model(data)
        
        if use_gpu:
            prediction = prediction.cpu().detach().numpy()
        else:
            prediction = prediction.detach().numpy()
        prediction = np.squeeze(prediction)
        basename = os.path.basename(file_)
        np.savetxt(os.path.join(target_root, basename),prediction)

if __name__ == "__main__":
    test_file_list = glob.glob(r"D:\Listener\DataDriven\All_test\All_test\sy\zhurong_sync\*_phoneme.txt")
    result_dir = "./test_Result/sp-nonPE-conv7-new-loss-130"
    inference(test_file_list,
            r"E:\torch学习\ConvSequenceRegression\new_loss_models\epoch_130_loss0.023453809320926666.pkl",
            result_dir)
    
    target_dir_padding = r"E:\torch学习\conv-regression\padding_result\sp-smooth-nonPE-conv7-new-loss-130"
    target_dir_euler = r"E:\torch学习\conv-regression\euler_result\sp-smooth-nonPE-conv7-new-loss-130"
    target_dir_siyuanshu = r"E:\torch学习\conv-regression\siyuanshu_result\sp-smooth-nonPE-conv7-new-loss-130"
    mkdir(target_dir_padding)
    mkdir(target_dir_euler)
    mkdir(target_dir_siyuanshu)
    for i in glob.glob(os.path.join(result_dir,"*.txt")):
        process_head_result(i,target_dir_padding)
        process_for_new(i, target_dir_euler)
        process_for_new_siyuanshu(i, target_dir_siyuanshu)
