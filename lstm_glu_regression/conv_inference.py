from model import Conv_LstmHead,GatedLinearUnit
from utils import input_phoneme_file,process_head_result,process_for_new,process_for_new_siyuanshu,load_model_path
from utils import input_phoneme_file_with_silence
import torch.nn as nn
import torch
import os
import glob
import Hyperparams as hp
import numpy as np
def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)

class inference_model(nn.Module):
    def __init__(self,glu_layer_size,lstm_layer_size):
        super(inference_model, self).__init__()
        self.embed_data = nn.Embedding(hp.vocab_size, hp.embedding_dim)
        self.glu_list = nn.ModuleList([])
        self.glu_list.extend([GatedLinearUnit(7, hp.embedding_dim,hp.embedding_dim*2) for i in range(glu_layer_size)])
        self.lstm_layers = nn.LSTM(hp.embedding_dim,hp.hidden_dim,lstm_layer_size,batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(hp.hidden_dim,hp.target_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):

        # expand_dims
        out = self.embed_data(x)
        out = torch.transpose(out,1,2)

        for glu in self.glu_list:
            out = glu(out)

        out = torch.transpose(out,1,2)
        lstm_out,_= self.lstm_layers(out)
        out = lstm_out + out
        out = self.output(out)
        return out
def inference(test_file_list,model_path,target_root):
    mkdir(target_root)
    file_list = []
    file_list.extend(test_file_list)
    model = inference_model(2,1)
    model.load_state_dict(torch.load(model_path))
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("use cuda")
        model = model.cuda()
    model.eval()
    for file_ in file_list:
        data = input_phoneme_file_with_silence(file_)
        data = np.array(data).astype("int")
       
        data = torch.from_numpy(data)
       
        data = torch.unsqueeze(data, 0)
        
        data = data.long()
        if use_gpu:
            data = data.cuda()
            
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
    result_dir = "./test_Result/conv_sp"
    # model_path, start_index = load_model_path(hp.save_dir)
    inference(test_file_list,
                r"D:\Listener\DataDriven\danlu\silence-lstm-head\models\conv_lstm\epoch_440_loss0.022954383864998817.pkl",
                result_dir)
    
    target_dir_padding = r"E:\torch学习\lstm\conv_sp\padding_result_0229"
    target_dir_euler = r"E:\torch学习\lstm\conv_sp\euler_result_0229"
    target_dir_siyuanshu = r"E:\torch学习\lstm\conv_sp\siyuanshu_result_0229"
    # mkdir(target_dir_padding)
    # mkdir(target_dir_euler)
    # mkdir(target_dir_siyuanshu)
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