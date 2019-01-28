import os
vocab_size = 1593
embedding_dim = 256
hidden_dim = 256
target_dim = 3
max_len = 550
pe = False
learning_rate = 0.0001
continue_train = False
batch_size = 64
save_dir = r"E:torch学习\ConvSequenceRegression\new_loss_models"
log_dir = r"E:torch学习\ConvSequenceRegression\new_loss_logs"
for i in [save_dir,log_dir]:
    if not os.path.exists(i):
        os.makedirs(i)