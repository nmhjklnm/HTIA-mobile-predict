import heapq
import torch
from torch import nn
import torch.nn.functional as F
import config
import random
from decoder.attention import Attention
from data.num_sequence import Num_sequence


class NumDecoder(nn.Module):
    def __init__(self):
        super(NumDecoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=config.num_embedding,
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.num_sequence.PAD)
        self.gru = nn.GRU(input_size=config.embedding_dim,
                          hidden_size=config.hidden_size,
                          num_layers=config.num_layer,
                          batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.num_embedding)  # 输入hidden大小，输出 所有词的种类数

        self.attn =Attention(config.method, config.hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.wa=nn.Linear(config.hidden_size*2,config.hidden_size,bias=False)  #输入decoder_hidden_size+encoder_hidden_size输出decoder
    def forward(self, encoder_hidden,encoder_outputs,label,numpairs):

        '''
        @param label: [batch_size, max_len+1]
        @param encoder_hidden: [1, batch_size, hidden_size]
        @param encoder_outputs: [batch_size,seq_len,hidden_size]
        @return:
        '''
        '''
            因为要告诉解码器开始解码，且我们定义只要遇到SOS就表示句子的开始，且每一个时间步(词)都要单独计算一次，不同于编码器在最后一步才生成结果
            因此形状要和batch_size相同，故为[batch_size,1]，即batch_size个数据
            torch.LongTensor([[SOS]]*batch_size)
        '''
        # 1.获取encoder的输出，作为decoder第一次的hidden_state
        # 2.准备decoder第一个时间步的输入: [batch_size, 1] SOS作为输入
        # 3.在第一个时间步上进行计算，得到第一个时间步的输出 和 hidden_state
        # 4.把前一个时间步的输出进行计算，得到第一个最后的输出结果
        # 5.把前一次的hidden_state 作为当前时间步的hidden_state的输入，把前一次的输出，作为当前时间步的输入
        # 6.循环4-5步骤
        batch_size = encoder_hidden.size(1)
        decoder_hidden = encoder_hidden
        decoder_input = torch.LongTensor([[config.num_sequence.SOS]]*batch_size).to(config.device)
        '''
            因为是对 label进行操作，因为我们的input和label的要求是: label就是在input后面加个0，所以label的长度比input要多1，故max_len+1
            input: 123
            lable: 1230
            PS: [batch_size, config.max_len+1, config.num_embedding] <==> [batch_size, max_len+1, vocab_size]
        '''
        decoder_outputs= torch.zeros([batch_size,config.day_walk_max+1, config.num_embedding]).to(config.device)
        for t in range(config.day_walk_max+1):  # 这里的max_len+1同上理
            decoder_output_week = torch.zeros([batch_size, 7, config.hidden_size]).to(config.device) #放在外循环，因为每一时间步的输出都得初始化一个week
            for i in range(7):
                encoder_outputs_day=encoder_outputs[:, i * 20:20 * i + 20, :]
                numpair=numpairs[:, i * 20:20 * i + 20] #取出每天的长度，用来mask  [batch_size,seq_len]
                attn_weights_day=self.attn(decoder_hidden,encoder_outputs_day,numpair)
                decoder_output_day=torch.bmm(attn_weights_day,encoder_outputs_day)
                decoder_output_week[:,i:i+1,:]=decoder_output_day
            # decoder_output_t->[batch_size,vocal_size]
            decoder_output_t, decoder_hidden, attn_weights_week = self.forward_step_attn(decoder_input, decoder_hidden,decoder_output_week)
            decoder_outputs[:,t:t+1,:]=decoder_output_t.unsqueeze(1)
            if random.random() > config.teacher_forcing_ratio:
                decoder_input = label[:, t].unsqueeze(-1)
            else:
                value, index = torch.topk(decoder_output_t, 1)  #value [batch_size,1]
                decoder_input = index
        return decoder_outputs, decoder_input

    def forward_step_attn(self, decoder_input, decoder_hidden, coder_outputs):
        """
        @param decoder_input:[batch_size,1]
        @param decoder_hidden:
        @param coder_outputs:
        @return:
        """
        embeded = self.dropout(self.embedding(decoder_input))  # embeded: [batch_size,1 , embedding_dim]
        # TODO 可以把embeded的结果和前一次的context（初始值为全0tensor） concate之后作为结果
        # rnn_input = torch.cat((embeded, last_context.unsqueeze(0)), 2)

        # gru_out:[256,1, 128]  decoder_hidden: [1, batch_size, hidden_size]
        gru_out, decoder_hidden = self.gru(embeded, decoder_hidden)
        gru_out = gru_out.squeeze(1)
        # TODO 注意：如果是单层，这里使用decoder_hidden没问题（output和hidden相同）
        # 如果是多层，可以使用GRU的output作为attention的输入
        # 开始使用attention
        attn_weights = self.attn(decoder_hidden.squeeze(0), coder_outputs,numpair=None)  # attn_weights [batch_size,1,seq_len] * [batch_size,seq_len,hidden_size]
        context = attn_weights.bmm(coder_outputs)                # [batch_size,1,hidden_size]

        gru_out = gru_out.view(-1,config.hidden_size)  # [batch_size,hidden_size]
        context = context.squeeze(1)  # [batch_size,hidden_size]
        # 把output和attention的结果合并到一起
        concat_input = torch.cat((gru_out, context), 1)  # [batch_size,hidden_size*2]
        concat_output = torch.tanh(self.wa(concat_input))  # [batch_size,hidden_size]

        output = F.log_softmax(self.fc(concat_output), dim=-1)  # [batch_Size, vocab_size]
        # out = out.squeeze(1)
        return output, decoder_hidden,attn_weights

    def evaluatoin_beamsearch_heapq(self, encoder_outputs, encoder_hidden,numpairs,topk):
        """使用 堆 来完成beam search，对是一种优先级的队列，按照优先级顺序存取数据"""

        batch_size = encoder_hidden.size(1)
        # 1. 构造第一次需要的输入数据，保存在堆中
        decoder_input = torch.LongTensor([[Num_sequence.SOS] * batch_size]).to(config.device)
        decoder_hidden = encoder_hidden  # 需要输入的hidden
        attention_week, attention_day=0,[] #初始化 用来存储记录注意力数据
        assert batch_size == 1, "beam search的过程中，batch_size只能为1"
        prev_beam = Beam(topk)
        prev_beam.add(1, False, [decoder_input], decoder_input, decoder_hidden,attention_week,attention_day)
        while True:
            cur_beam = Beam(topk=topk)
            # 2. 取出堆中的数据，进行forward_step的操作，获得当前时间步的output，hidden
            # 这里使用下划线进行区分
            for _probility, _complete, _seq, _decoder_input, _decoder_hidden ,attention_week,attention_day in prev_beam:
                # 判断前一次的_complete是否为True，如果是，则不需要forward
                # 有可能为True，但是概率并不是最大
                if _complete == True:
                    cur_beam.add(_probility, _complete, _seq, _decoder_input, _decoder_hidden,attention_week,attention_day)
                else:
                    decoder_output_week = torch.zeros([batch_size, 7, config.hidden_size]).to(config.device)  # 放在外循环，因为每一时间步的输出都得初始化一个week
                    for i in range(7):
                        numpair = numpairs[:, i * 20:20 * i + 20]  # 取出每天的长度，用来mask
                        encoder_outputs_day = encoder_outputs[:, i * 20:20 * i + 20, :]
                        attn_weights_day = self.attn(decoder_hidden, encoder_outputs_day,numpair)
                        decoder_output_day = torch.bmm(attn_weights_day, encoder_outputs_day)
                        decoder_output_week[:, i:i + 1, :] = decoder_output_day
                        attention_day.append(attn_weights_day) #记录天注意力
                    # decoder_output_t->[batch_size,vocal_size]
                    decoder_output_t, decoder_hidden, attention_week = self.forward_step_attn(_decoder_input.T,
                                                                                                 _decoder_hidden,
                                                                                                 decoder_output_week)
                    value, index = torch.topk(decoder_output_t, topk)  # [batch_size=1,beam_widht=3]
                    # 3. 从output中选择topk（k=beam width）个输出，作为下一次的input
                    for m, n in zip(value[0], index[0]):
                        decoder_input = torch.LongTensor([[n]]).to(config.device)  #[batch_size,
                        seq = _seq + [n]  # 加上新序列
                        probility = _probility +m  #概率相乘

                        complete =  n.item() == Num_sequence.EOS

                        # 4. 把下一个实践步骤需要的输入等数据保存在一个新的堆中
                        cur_beam.add(probility, complete, seq,decoder_input, decoder_hidden,attention_week,attention_day)
                        # 5. 获取新的堆中的优先级最高（概率最大）的数据，判断数据是否是EOS结尾或者是否达到最大长度，如果是，停止迭代 #

            condition1= [ i for _, i, j, _, _, attention_week, attention_day in cur_beam.pop_all()]
            condition2= [ len(j) for _, i, j, _, _, attention_week, attention_day in cur_beam.pop_all()]
            if all(condition1) or max(condition2)==21:  # 减去sos
                temp=[[self._prepar_seq(i),attention_week,attention_day] for _,_,i,_,_,attention_week,attention_day in cur_beam.pop_all()]
                return temp
            else:
                # 6. 则重新遍历新的堆中的数据
                prev_beam = cur_beam

    def _prepar_seq(self,seq):#对结果进行基础的处理，共后续转化为文字使用
        if  seq[0].item() == Num_sequence.SOS:
            seq=seq[1:]
        #先不裁EOS
        if  seq[-1].item() == Num_sequence.EOS:
            seq = seq[:-1]
        seq = [i.item() for i in seq]
        return seq

class Beam:
    def __init__(self,topk):
        self.heap = list() #保存数据的位置
        self.beam_width = topk #保存数据的总数

    def add(self,probility,complete,seq,decoder_input,decoder_hidden,attention_week,attention_day):
        """
        添加数据，同时判断总的数据个数，多则删除
        :param probility: 概率乘积
        :param complete: 最后一个是否为EOS
        :param seq: list，所有token的列表
        :param decoder_input: 下一次进行解码的输入，通过前一次获得
        :param decoder_hidden: 下一次进行解码的hidden，通过前一次获得
        :param attention_week 周注意力机制
        :param attention_day天注意力机制
        :return:
        """
        heapq.heappush(self.heap,[probility,complete,seq,decoder_input,decoder_hidden,attention_week,attention_day])
        #判断数据的个数，如果大，则弹出。保证数据总个数小于等于3
        if len(self.heap)>self.beam_width:
            heapq.heappop(self.heap)
    def pop_all(self):
        """
        @return:取出其中所有值，按概率大到小排序
        """
        return heapq.nlargest(self.beam_width,self.heap)

    def __iter__(self):#让该beam能够被迭代
        return iter(self.heap)