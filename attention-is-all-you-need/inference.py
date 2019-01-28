from transformer.Modules import Encoder
from utils import input_phoneme_file,process_head_result,process_for_new,process_for_new_siyuanshu,load_model_path
from utils import input_phoneme_file_with_silence
import torch
import os
import glob
import Hyperparams as hp
import numpy as np
import pdb

def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)

def inference(test_file_list,model_path,target_root, word_dict):
    mkdir(target_root)
    file_list = []
    file_list.extend(test_file_list)
    model = Encoder(hp)
    model.load_state_dict(torch.load(model_path))
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use cuda")
        model = model.cuda()
    model.eval()
    for file_ in file_list:
        data = input_phoneme_file_with_silence(file_, word_dict)
        
        length = len(data)

        pos = np.arange(length) + 1
        pos_idx = torch.from_numpy(pos)
        pos_idx = torch.unsqueeze(pos_idx, 0)

        data = np.array(data).astype("int")
        # print(data)
        data = torch.from_numpy(data)
       
        data = torch.unsqueeze(data, 0)
        
        data = data.long()
        pos_idx = pos_idx.long()
        if use_gpu:
            data = data.cuda()
            pos_idx = pos_idx.cuda()
        print(data)    
        prediction = model(data, pos_idx)
        print(prediction)
        if use_gpu:
            prediction = prediction.cpu().detach().numpy()
        else:
            prediction = prediction.detach().numpy()
        prediction = (np.squeeze(prediction) + 1) / 2
        basename = os.path.basename(file_)
        np.savetxt(os.path.join(target_root, basename),prediction)

if __name__ == "__main__":
    test_file_list = glob.glob(r"D:\Listener\DataDriven\All_test\All_test\sy\zhurong_sync\*_phoneme.txt")
    result_dir = "./test_Result/sp"
    word_dict = "./json/9月数据(眼部骨骼都有)/DataSetFullSilenceWordDict.json"
    # model_path, start_index = load_model_path(hp.save_dir)
    inference(test_file_list,
                r"E:\torch学习\code\sequence\attention-is-all-you-need-reconstruct\checkpoints\smooth_models\epoch_100_loss0.4169119596481323.pkl",
                result_dir, word_dict)
    
    target_dir_padding = r"E:\torch学习\result\self-attention-smooth\sp\padding_result"
    target_dir_euler = r"E:\torch学习\result\self-attention-smooth\sp\euler_result"
    target_dir_siyuanshu = r"E:\torch学习\result\self-attention-smooth\sp\siyuanshu_result"
    mkdir(target_dir_padding)
    mkdir(target_dir_euler)
    mkdir(target_dir_siyuanshu)
    for i in glob.glob(os.path.join(result_dir,"*.txt")):
        # process_head_result(i,target_dir_padding)
        # process_for_new(i, target_dir_euler)
        # process_for_new_siyuanshu(i, target_dir_siyuanshu)
        gain = 1
        process_head_result(result_file=i,smooth=False,gain=1,target_root=target_dir_padding+"_non_smooth_{}".format(gain))
        process_for_new(result_file=i,smooth=False,gain=1, target_root=target_dir_euler+"_non_smooth_{}".format(gain) )
        process_for_new_siyuanshu(result_file = i,smooth=False,gain=1, target_root=target_dir_siyuanshu+"_non_smooth_{}".format(gain))
        process_head_result(result_file=i,smooth=True,gain=1,target_root=target_dir_padding+"_smooth_{}".format(gain))
        process_for_new(result_file=i,smooth=True,gain=1, target_root=target_dir_euler+"_smooth_{}".format(gain))
        process_for_new_siyuanshu(result_file = i,smooth=True,gain=1, target_root=target_dir_siyuanshu+"_smooth_{}".format(gain))
        gain=0.5
        process_head_result(result_file=i,smooth=True,gain=gain,target_root=target_dir_padding+"_smooth_{}".format(gain))
        process_for_new(result_file=i,smooth=True,gain=gain, target_root=target_dir_euler+"_smooth_{}".format(gain))
        process_for_new_siyuanshu(result_file = i,smooth=True,gain=gain, target_root=target_dir_siyuanshu+"_smooth_{}".format(gain))
        process_head_result(result_file=i,smooth=False,gain=gain,target_root=target_dir_padding+"_nonsmooth_{}".format(gain))
        process_for_new(result_file=i,smooth=False,gain=gain, target_root=target_dir_euler+"_nonsmooth_{}".format(gain))
        process_for_new_siyuanshu(result_file = i,smooth=False,gain=gain, target_root=target_dir_siyuanshu+"_nonsmooth_{}".format(gain))