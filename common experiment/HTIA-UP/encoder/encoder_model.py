
import torch
from torch import nn
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import config
from embedding_concat import embedding_concat



class NumEncoder(nn.Module):
    def __init__(self):
        super(NumEncoder, self).__init__()
        self.embedding_traj= nn.Embedding(num_embeddings=config.num_embedding,
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.num_sequence.PAD)
        self.embedding_weeki=nn.Embedding(num_embeddings=config.num_embedding,
                                      embedding_dim=config.attr_embedding_dim)
        self.embedding_gap=nn.Embedding(num_embeddings=config.num_embedding,
                                      embedding_dim=config.attr_embedding_dim)

        # 指定 PAD 不需要进行更新，这里需要传入的是数值
        self.gru = nn.GRU(input_size=config.embedding_dim+2*config.attr_embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layer,
                          batch_first=True)
        self.dropout=nn.Dropout(0.5)
        self.line=nn.Linear(config.embedding_dim+2*config.attr_embedding_dim,config.hidden_size)
    def forward(self,numpairs,attr):
        """

        @param numpairs:
        @param attr:
        @return:
        """

        # TODO 可以考虑打包加速这个过程
        attr_weeki,attr_gap=attr[:,:,0],attr[:,:,1]
        embeded_traj = self.embedding_traj(numpairs)# [batch_size, max_len, embedding_dim]
        embeded_attr_weeki=self.embedding_weeki(attr_weeki)
        embeded_gap=self.embedding_gap(attr_gap)
        embeded_attr = self.dropout(torch.concat((embeded_attr_weeki, embeded_gap), 2))
        # TODO 构建轨迹和属性的拼接向量
        embeded=embedding_concat(embeded_traj,embeded_attr) #合并后的276要转化为256的
        embededsss=self.line(embeded)
        # embeded = pack_padded_sequence(embeded, input_length, batch_first=True)  # 对embeded结果进行打包，加速更新速度
        output, hidden = self.gru(embeded)  # hidden:[num_layer*num_directions, batch_size, hidden_size] [1, 128, 64]
        # output, output_length = pad_packed_sequence(output, batch_first=True, padding_value=config.num_sequence.PAD)  # output:[batch_size, max_len, num_directions*hidden_size] [128, 9, 1*64]

        return embededsss, hidden


if __name__ == '__main__':
    from data.dataset import train_dataloader
    numencoder = NumEncoder()
    print(numencoder)
    config.num_embedding=5000#用于测试，取一个随机大的数，否则，numembedding太小没法跑embedding
    for numpairs,attr,label in train_dataloader:
        output, hidden = numencoder(numpairs,attr)

        break