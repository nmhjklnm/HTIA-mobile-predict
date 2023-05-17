
import torch

from data.num_sequence import Num_sequence
import os
import pickle
txtpath="raw_data\\981762.txt"
data=txtpath[-10:-4]
if os.path.exists("dic\\" +data + '.pkl') == False:
    num_sequence = Num_sequence()
    pickle.dump(num_sequence, open(f"dic\{data}.pkl", 'wb'))
else:
    num_sequence=pickle.load(open(f'dic\{data}.pkl','rb'))
    print('导入预设字典完成')

learning_rate=1e-3
# 根据cuda是否可用，自动选择训练的设备
# device='cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 模型保存
data=txtpath[9:15]
manu_seed=0

#s版本用来加dropout ，sm版本用来做消融实验
model_save_path = f'modelsm/{data}/seq2seq_{manu_seed}.pt'
optimizer_save_path = f'modelsm/{data}/optimizer_{manu_seed}.pt'

# 模型配置
train_coll=0.9
batch_size =int(num_sequence.sample_len*train_coll/30)# 训练  5或20
test_batch_size=1
num_embedding = len(num_sequence)  # 词典中不同种类词的个数


attr_embedding_dim=20
# GRU
num_layer = 1
hidden_size = 256

embedding_dim = 256    # 词向量的维度
# decoder
teacher_forcing_ratio = 0.5
method='concat'

day_walk_max=num_sequence.sample_day_walk
