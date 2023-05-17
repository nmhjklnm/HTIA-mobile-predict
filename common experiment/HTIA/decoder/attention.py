from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import math

class Attention(nn.Module):

    def __init__(self, method, hidden_size):

        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        assert self.method in ["dot", "general", "concat"], "method only dot,general,concat,now is{}".format(
            self.method)

        if self.method == "dot":
            pass
        elif self.method == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.method == "concat":
            self.Qa = nn.Linear(hidden_size*2, hidden_size, bias=False)
            self.Va = nn.Linear(hidden_size, 1,bias=False)


    def forward(self, hidden, encoder_outputs,numpair=None):

        hidden = hidden.view(-1,hidden.size(-1))  # [batch_size,hidden_size]

        if self.method == "dot":
            return self.dot_score(hidden, encoder_outputs)
        elif self.method == "general":

            return self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            return self.concat_score(hidden, encoder_outputs,numpair)

    def dot_score(self, hidden, encoder_outputs):



        hidden = hidden.unsqueeze(-1)
        attn_energies = torch.bmm(encoder_outputs, hidden)
        attn_energies = attn_energies.squeeze(-1)  # [batch_size,seq_len,1] ==>[batch_size,seq_len]

        return F.softmax(attn_energies,dim=-1).unsqueeze(1)  # [batch_size,1,seq_len]

    def general_score(self, hidden, encoder_outputs):

        x = self.Wa(hidden)
        x = x.view(x.size(0),x.size(1),1)
        attn_energies = torch.bmm(encoder_outputs, x).squeeze(-1)
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)

    def concat_score(self, hidden, encoder_outputs,numpair=None):

        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.Qa(torch.cat((hidden,encoder_outputs), dim=2)))
        attention = self.Va(energy).squeeze(2)
        if numpair is not None:
            attention = attention.masked_fill(numpair==0, -1e10)
        return F.softmax(attention, dim=1).unsqueeze(1)

class Self_Attention(nn.Module):
    def __init__(self,method,input_vector_dim=None,dim_k=None,dim_v=None,h=None):

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
            self._norm_fact = 1 / np.sqrt(dim_k)

            self.gamma = nn.Parameter(torch.zeros(1))
        elif self.method == "mult_self":
            assert input_vector_dim % h == 0 ,"head divided input_vector_dim completely"

            self.dim_split_head = input_vector_dim // h
            self.h = h
            # 定义W^q, W^k, W^v和W^o矩阵这四个矩阵，简化为同型矩阵
            self.W_q = nn.Linear(input_vector_dim, input_vector_dim, bias=False)
            self.W_k = nn.Linear(input_vector_dim, input_vector_dim, bias=False)
            self.W_v = nn.Linear(input_vector_dim, input_vector_dim, bias=False)
            self.W_o = nn.Linear(input_vector_dim,input_vector_dim, bias=False)

    def forward(self,x,mask=None):

        x=x.to("cuda:0")
        if self.method == "self":
            return self.self_attention(x,mask)
        elif self.method == "mult_self":
            return self.mult_self_attention(x,mask)


    def self_attention(self,x,mask=None):

        Q = self.W_q(x)
        K = self.W_k(x).transpose(1,2)
        V = self.W_v(x)

        attn=torch.bmm(Q, K) * self._norm_fact
        if mask is not None:
            mask=torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-1).transpose(-1,-2))
            attn = attn.masked_fill(mask == 0, -1e10)
        attn = nn.Softmax(dim=-1)(attn)

        output = torch.bmm(attn, V)

        return output ,attn

    def mult_self_attention(self,x,mask=None):

        batch_size = x.size(0)

        query = self.W_q(x).view(batch_size, -1, self.h, self.dim_split_head).transpose(1, 2)
        key   = self.W_k(x).view(batch_size, -1, self.h, self.dim_split_head).transpose(1, 2)
        value = self.W_v(x).view(batch_size, -1, self.h, self.dim_split_head).transpose(1, 2)


        attn = (torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim_split_head))


        if mask is not None:

            mask = torch.bmm(mask.unsqueeze(-1).to(torch.float64), mask.unsqueeze(-1).transpose(-1, -2).to(torch.float64))
            mask=mask.unsqueeze(1).repeat(1,attn.size(1),1,1)

            attn = attn.masked_fill(mask == 0, -1e10)  # 输入是有值的输入矩阵，就不用转化为布尔矩阵，这里直接能转化

        attn=attn.softmax(dim=-1)

        x=torch.matmul(attn, value)




        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.h * self.dim_split_head)
        )

        return self.W_o(x) ,attn



if __name__ == "__main__":
    x=torch.randn([128,20,256])
    mask=torch.cat((torch.randn([128,13]),torch.zeros([128,7])),dim=-1)
    # attn=Self_Attention(method="self",input_vector_dim=256,h=8)
    attn = Self_Attention(method="mult_self", input_vector_dim=256, h=8)
    a=attn(x,mask)
    print(a)





