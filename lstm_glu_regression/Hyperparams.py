import os
vocab_size = 1625
embedding_dim = 256
hidden_dim = 256
num_layers = 2
target_dim = 3
max_len = 550
pe = False
learning_rate = 0.0001
continue_train = True
batch_size = 64
save_dir = "/data/head-models/conv-lstm-silence/sp"
log_dir = "/data/head-logs/conv-lstm-silence/sp"
for i in [save_dir,log_dir]:
    if not os.path.exists(i):
        os.makedirs(i)
