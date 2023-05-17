import torch
from seq2seq.seq2seq_model import Seq2Seq
from data.dataset import test_dataloader
import config
from tqdm import tqdm
import numpy as np
from data.num_sequence import Num_sequence
import Levenshtein
import seaborn as sns; sns.set()

def eval(decoder_outputs,label):
    value, index = torch.topk(decoder_outputs,1)
    #从tensor裁剪到list
    results=[]
    for i in range(index.size(0)):
        indices =_prepar_seq(index[i])
        labels=_prepar_seq(label[i])
        #指标构建
        Mre,Ma,Mr=Mre_Ma_Mr(indices,labels)
        results.append([Mre,Ma,Mr])
    a=np.average(np.array(results), axis=0)  # 按列求均值
    return a

def eval_in_train(seq2seq,topk):
    print(f"beam search topk:{topk}")
    seq2seq.eval()
    results = []  # 存放所有的预测结果
    with torch.no_grad():
        for numpairs,attr,label in test_dataloader:
            numpairs = numpairs.to(config.device)
            attr = attr.to(config.device)
            indices = seq2seq.evaluate(numpairs, attr,topk=topk)
            temp=[]
            label=_prepar_seq(label)
            for i in indices:
                Mre,Ma,Mr=Mre_Ma_Mr(i[0],label)
                temp.append([Mre,Ma,Mr,i[1],i[2]])
            a=sorted(temp,key=lambda x:x[0])[0]
            results.append(a)

    a=np.average(np.array([[i[0:3]] for i in results]).squeeze(1), axis=0)  # 按列求均值
    print("eval_in_train",a)
    return a

def eval_beam_search(seq2seq,topk):
    seq2seq.load_state_dict(torch.load(config.model_save_path))  # 加载模型
    seq2seq.eval()
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), ascii=True)
    bar.set_description('eval')
    results = []  # 存放所有的预测结果

    with torch.no_grad():
        for index, (numpairs,attr,label) in bar:
            import pickle
            # with open("attn/envirorment.pickle", "wb") as file:
            #     pickle.dump((numpairs,attr,label), file)
            numpairs = numpairs.to(config.device)
            attr = attr.to(config.device)
            indices ,self_attn= seq2seq.evaluate(numpairs, attr,topk=topk)
            temp=[]
            label=_prepar_seq(label)
            for i in indices:
                Mre,Ma,Mr=Mre_Ma_Mr(i[0],label)
                temp.append([Mre,Ma,Mr,i[1],i[2],self_attn,(numpairs,attr,label)])
            a=sorted(temp,key=lambda x:x[0])[0]
            results.append(a)

    a=np.average(np.array([[i[0:3]] for i in results]).squeeze(1), axis=0)  # 按列求均值
    return a


def Mre_Ma_Mr(indices,label):
    Mre=(Levenshtein.distance(indices,label))/( len(indices) if len(indices)>len(label) else len(label))
    Ma=sum([1 if i in indices else 0 for i in indices[:len(label)]])/ len(indices)
    Mr=sum([1 if i in indices else 0 for i in indices[:len(label)]]) / len(label)
    return Mre,Ma,Mr

def _prepar_seq(seq):
    seq=seq.view(-1)
    if  seq[0].item() == Num_sequence.SOS:
        seq=seq[1:]

    seq = [i.item() for i in seq if i.item()>3]
    return seq
if __name__ == '__main__':
    seq2seq = Seq2Seq().to(config.device) # 实例化模型

    seq2seq.load_state_dict(torch.load(config.model_save_path))  # 加载模型
    print("based beam_search eval",10*"*")
    seq2seq.modules()
    print(eval_beam_search(seq2seq,3))