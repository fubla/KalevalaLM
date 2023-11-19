# train model using blocks of size block_size
block_size = 256  # max context length
batch_size = 64
max_iters = 5000
eval_interval = 200
lr = 1e-4
device = 'cuda'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
