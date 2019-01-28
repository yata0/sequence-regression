from tensorboardX import SummaryWriter
import glob
import os
def write_summary(writer,l1_loss, smooth_loss, total_loss,iter_index):
    writer.add_scalar("loss/l1_loss",l1_loss,iter_index)
    writer.add_scalar("loss/smooth_loss",smooth_loss,iter_index)
    writer.add_scalar("loss/loss",total_loss,iter_index)

def load_model_path(save_path):
    pth_list = glob.glob(os.path.join(save_path, "*.pkl"))
    indexs = [int(os.path.basename(f).split("_")[1]) for f in pth_list]
    index = max(indexs)

    model_path = glob.glob(os.path.join(save_path,"epoch_{}*.pkl".format(index)))[0]
    return model_path, index