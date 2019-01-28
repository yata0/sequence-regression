import os
vocab_size = 1625
embedding_size = 256
hidden_dim = 256
num_layers = 2
target_dim = 3
max_len = 550
learning_rate = 0.0001
continue_train = False
batch_size = 16
dim_model = 256
dim_inner = 512
n_head = 4 
d_k = 64 
d_v = 64 
dropout = 0.4
save_dir = "./checkpoints/smooth0.01_models"
log_dir = "./checkpoints/smooth0.01_logs"

for i in [save_dir,log_dir]:
    if not os.path.exists(i):
        os.makedirs(i)
