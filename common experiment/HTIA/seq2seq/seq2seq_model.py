

from torch import nn
from encoder.encoder_model import NumEncoder
from decoder.decoder_model import NumDecoder
from decoder.attention import Self_Attention
import config

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = NumEncoder()
        self.attn = Self_Attention(method="mult_self", input_vector_dim=config.hidden_size, h=8)
        self.decoder = NumDecoder()

    def forward(self,numpairs,attr,label):
        '''
        @param numpairs: [batch_size, max_len]
        @return:
        '''
        # encoder_outputs: [batch_size, max_len, num_directions*hidden_size]
        # encoder_hidden: [num_layer*num_directions, batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(numpairs,attr)

        encoder_outputs,_= self.attn(encoder_outputs, numpairs)

        # decoder_outputs: [batch_size, max_len, vacab_size]
        # decoder_hidden: [num_layer*num_directions, batch_size, hidden_size]
        decoder_outputs, decoder_input = self.decoder(encoder_hidden,encoder_outputs,label,numpairs)
        return decoder_outputs, decoder_input

    def evaluate(self, numpairs, attr,topk):
        """
        decoder_outputs->[batch_size, max_len, vacab_size]
        @param numpairs:
        @param attr:
        @return:
        """
        encoder_outputs, encoder_hidden = self.encoder(numpairs,attr)
        encoder_outputs ,self_attn= self.attn(encoder_outputs, numpairs)
        return self.decoder.evaluatoin_beamsearch_heapq(encoder_outputs,encoder_hidden,numpairs,topk=topk),self_attn


if __name__=="__main__":
    import  config
    from data.dataset import train_dataloader
    model =Seq2Seq()
    config.num_embedding = 50  # 用于测试，取一个随机大的数，否则，numembedding太小没法跑embedding
    for numpairs, traj_day_len, attr, label in train_dataloader:
        output, hidden = model(numpairs, traj_day_len, attr,label)
        break