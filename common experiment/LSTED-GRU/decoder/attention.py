from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math

class Attention(nn.Module):

    def __init__(self, method, hidden_size):
        """
        这个attention主要用于基础attention机制
        :param method:
        :param hidden_size:
        """
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        assert self.method in ["dot", "general", "concat"], "method 只能是 dot,general,concat,当前是{}".format(
            self.method)

        if self.method == "dot":
            pass
        elif self.method == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.method == "concat":
            self.Qa = nn.Linear(hidden_size*2, hidden_size, bias=False)
            self.Va = nn.Linear(hidden_size, 1,bias=False)


    def forward(self, hidden, encoder_outputs,numpair=None):
        """
        :param hidden:[1,batch_size,hidden_size]
        :param encoder_outputs: [batch_size,seq_len,hidden_size]
        :return:
        """
        hidden = hidden.view(-1,hidden.size(-1))  # [batch_size,hidden_size]

        if self.method == "dot":
            return self.dot_score(hidden, encoder_outputs)
        elif self.method == "general":
            #TODO numpair暂时用不到mask先不添加了
            return self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            return self.concat_score(hidden, encoder_outputs,numpair)

    def dot_score(self, hidden, encoder_outputs):
        """
        dot attention
        :param hidden:[batch_size,hidden_size] --->[batch_size,hidden_size,1]
        :param encoder_outputs: [batch_size,seq_len,hidden_size]
        :return:
        """
        # hidden :[hidden_size] -->[hidden_size,1] ，encoder_output:[seq_len,hidden_size]

        hidden = hidden.unsqueeze(-1)
        attn_energies = torch.bmm(encoder_outputs, hidden)
        attn_energies = attn_energies.squeeze(-1)  # [batch_size,seq_len,1] ==>[batch_size,seq_len]

        return F.softmax(attn_energies,dim=-1).unsqueeze(1)  # [batch_size,1,seq_len]

    def general_score(self, hidden, encoder_outputs):
        """
        general attenion
        :param hidden: [batch_size,hidden_size]
        :param encoder_outputs: [batch_size,seq_len,hidden_size]
        :return:
        """
        x = self.Wa(hidden)  # [batch_size,hidden_size]
        x = x.view(x.size(0),x.size(1),1) # [batch_size,hidden_size,1]
        attn_energies = torch.bmm(encoder_outputs, x).squeeze(-1)  # [batch_size,seq_len,1]
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)  # [batch_size,1,seq_len]

    def concat_score(self, hidden, encoder_outputs,numpair=None):
        """

        @param hidden: [batch_size,hidden_size]
        @param encoder_outputs:  [batch_size,seq_len,hidden_size]
        @param numpair: 为空就不用进行掩码操作，空值就还得mask
        @return: # 需要先进行repeat操作，变成和encoder_outputs相同的形状,让每个batch有seq_len个hidden_size
        """
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  #batch_size,seq_len,hidden_size
        energy = torch.tanh(self.Qa(torch.cat((hidden,encoder_outputs), dim=2)))  #batch_size,seq_len,hidden_size
        attention = self.Va(energy).squeeze(2) #attention->[batch_size,seq_len]
        if numpair is not None:            #非空就执行masked_fill,也就是globa_attention
            attention = attention.masked_fill(numpair==0, -1e10) #两者同型，[batch_size,seq_len]
        return F.softmax(attention, dim=1).unsqueeze(1)  # [batch_size,1,seq_len]

class Self_Attention(nn.Module):
    def __init__(self,method,input_vector_dim=None,dim_k=None,dim_v=None,h=None):
        """
        :param method: 所采取的方法
        :param input_vector_dim:  输入的隐藏维度
        :param dim_k: key矩阵的size 为了能够矩阵相乘 能够保证，要求dim_k=dim_q  不设定就默认是input_vector_dim值了
        :param dim_v: v矩阵的size                                         不设定就默认是input_vector_dim值了
        :param h: 头的数量
        """
        self.method=method
        super().__init__()
        self.input_vector_dim=input_vector_dim
        dim_k = input_vector_dim if dim_k == None else dim_k
        dim_v = input_vector_dim if dim_v == None else dim_v

        assert self.method in ["self", "mult_self"], "method 只能是 self,mult_self 当前是{}".format(self.method)
        if self.method == "self":
            self.W_q = nn.Linear(input_vector_dim, dim_k, bias=False)
            self.W_k = nn.Linear(input_vector_dim, dim_k, bias=False)
            self.W_v = nn.Linear(input_vector_dim, dim_v, bias=False)
            self._norm_fact = 1 / np.sqrt(dim_k)  #用来标准化
            # gamma充当残差连接线
            self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        elif self.method == "mult_self":
            assert input_vector_dim % h == 0 ,"head要能被input_vector_dim整除"
            #简化一下都用input_vector_dim来表示向量维度
            self.dim_split_head = input_vector_dim // h
            self.h = h
            # 定义W^q, W^k, W^v和W^o矩阵这四个矩阵，简化为同型矩阵
            self.W_q = nn.Linear(input_vector_dim, input_vector_dim, bias=False)
            self.W_k = nn.Linear(input_vector_dim, input_vector_dim, bias=False)
            self.W_v = nn.Linear(input_vector_dim, input_vector_dim, bias=False)
            self.W_o = nn.Linear(input_vector_dim,input_vector_dim, bias=False)

    def forward(self,x,mask=None):
        """
        :param x: [batch_size,seq_len,hidden_size或者叫input_vector_dim]
        :param mask:输入掩码矩阵 [batch_size,seq_len]
        :return:
        """
        x=x.to("cuda:0")
        if self.method == "self":
            return self.self_attention(x,mask)
        elif self.method == "mult_self":
            return self.mult_self_attention(x,mask)


    def self_attention(self,x,mask=None):
        # 通过W_q, W_k, W_v矩阵计算出，Q,K,V
        # Q,K,V矩阵的size为 (batch_size, input_num, output_vector_dim)
        Q = self.W_q(x)
        K = self.W_k(x).transpose(1,2)
        V = self.W_v(x)
        # transpose将K的size由(batch_size, input_num, output_vector_dim)，变为(batch_size, output_vector_dim，input_num)
        # 0,1,2 代表各个元素的下标，即变换前，batch_size所在的位置是0，input_num所在的位置是1
        attn=torch.bmm(Q, K) * self._norm_fact #attn->[batch_size, seq_len,seq_len] #
        if mask is not None:  # 非空就执行masked_fill,也就是globa_attention
            mask=torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-1).transpose(-1,-2))
            attn = attn.masked_fill(mask == 0, -1e10)  # 输入是有值的输入矩阵，就不用转化为布尔矩阵，这里直接能转化
        attn = nn.Softmax(dim=-1)(attn)
        # 最后再乘以 V
        output = torch.bmm(attn, V)

        return output ,attn

    def mult_self_attention(self,x,mask=None):
        """
        :@param x:[batch_size,seq_len,input_vector_dim]
        :return: 返回是attention后的结果，不是权
        """
        batch_size = x.size(0)
        #1. 求出Q, K, V，这里是求MultiHead的Q,K,V，所以Shape为(batch, head数, 词数，input_vector_dim/head数)
        query = self.W_q(x).view(batch_size, -1, self.h, self.dim_split_head).transpose(1, 2)
        key   = self.W_k(x).view(batch_size, -1, self.h, self.dim_split_head).transpose(1, 2)
        value = self.W_v(x).view(batch_size, -1, self.h, self.dim_split_head).transpose(1, 2)


        #2.执行QK^T / √dim_split_head,然后转化为概率
        # prarm attn:[batch_size,head,seq_len,seq_len] 这里的attn也就是权
        # [batch_size,head,seq_len,seq_len]*[batch_size,head,seq_len,input_vector_dim/head]=[batch_size,head,seq_len,input_vector_dim/head]
        attn = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_split_head))

        # 插入mask 此时的attn传入的[batch_size,head,seq_len]
        if mask is not None:  # 非空就执行masked_fill,也就是globa_attention
            # "baddbmm_cuda" not implemented for 'Long'
            mask = torch.bmm(mask.unsqueeze(-1).to(torch.float64), mask.unsqueeze(-1).transpose(-1, -2).to(torch.float64))
            mask=mask.unsqueeze(1).repeat(1,attn.size(1),1,1)
            # 复制8份 然后 (A*AT)
            attn = attn.masked_fill(mask == 0, -1e10)  # 输入是有值的输入矩阵，就不用转化为布尔矩阵，这里直接能转化

        attn=attn.softmax(dim=-1)
        x=torch.matmul(attn, value)

        #3. 多头合并，即将x的shape由(batch, head数, 词数，input_vector_dim/head数)，再变为 (batch, 词数，input_vector_dim）
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.h * self.dim_split_head)
        )
        # 4，最终通过W^o矩阵再执行一次线性变换，得到最终结果。
        return self.W_o(x)



if __name__ == "__main__":
    x=torch.randn([128,20,256])
    mask=torch.cat((torch.randn([128,13]),torch.zeros([128,7])),dim=-1)
    # attn=Self_Attention(method="self",input_vector_dim=256,h=8)
    attn = Self_Attention(method="mult_self", input_vector_dim=256, h=8)
    a=attn(x,mask)
    print(a)





