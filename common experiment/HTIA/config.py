
import torch
import os
import pickle

from data.num_sequence import Num_sequence

txtpath="raw_data\981762.txt"
data=txtpath[-10:-4]
manu_seed=2

if os.path.exists("dic\\" +data + '.pkl') == False:
    num_sequence = Num_sequence()
    pickle.dump(num_sequence, open(f"dic\{data}.pkl", 'wb'))
else:
    num_sequence=pickle.load(open(f'dic\{data}.pkl','rb'))
    print('The default dictionary is imported')
learning_rate=1e-3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型保存

model_save_path = f'models/{data}/seq2seq_{manu_seed}.pt'
optimizer_save_path = f'models/{data}/optimizer_{manu_seed}.pt'

# 模型配置
train_coll=0.9
batch_size =int(num_sequence.sample_len*train_coll/30)
test_batch_size=1
num_embedding = len(num_sequence)


attr_embedding_dim=20
# GRU
num_layer = 1
hidden_size = 256

embedding_dim = 256
# decoder
teacher_forcing_ratio = 0.5
method='concat'

day_walk_max=num_sequence.sample_day_walk
