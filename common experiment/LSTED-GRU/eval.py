
import torch
from seq2seq.seq2seq_model import Seq2Seq
from data.dataset import test_dataloader
import config
from tqdm import tqdm
import numpy as np
from num_sequence import Num_sequence
import Levenshtein
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
# 测试流程
# 1.准备测试数据
# 2.加载模型
# 3.获取预测值
# 4.反序列化，观察结果
#因为用了beam search 所以这里不得不一个一个测试
# seq2seq = Seq2Seq().to(config.device) # 实例化模型

# seq2seq.load_state_dict(torch.load(config.model_save_path,map_location='cpu'))  # 加载模型
# seq2seq.load_state_dict(torch.load(config.model_save_path))  # 加载模型

# def eval():
#     seq2seq.load_state_dict(torch.load(config.model_save_path))  # 加载模型
#     seq2seq.eval()
#     bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), ascii=True)
#     bar.set_description('eval')
#     results = []  # 存放所有的预测结果
#     # labels = []
#     with torch.no_grad(): #其实有了model_eval 也可以不用no_grad,用了就能速度更快一点
#         for index, (numpairs,attr,label) in bar:
#             numpairs = numpairs.to(config.device)
#             attr = attr.to(config.device)
#             decoder_outputs,_= seq2seq(numpairs,attr,label)
#             value, index = torch.topk(decoder_outputs,1)
#             #从tensor裁剪到list
#             indices =_prepar_seq(index)
#             label=_prepar_seq(label)
#             #指标构建
#             Mre,Ma,Mr=Mre_Ma_Mr(indices,label)
#             results.append([Mre,Ma,Mr])
#
#     a=np.average(np.array(results), axis=0)  # 按列求均值
#     print(a)
#     dd=pd.DataFrame(np.array(results),columns=["Mre","Ma","Me"])
#     sns.lineplot(data=dd)
#     plt.show()

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
    with torch.no_grad(): #其实有了model_eval 也可以不用no_grad,用了就能速度更快一点
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
    # labels = []
    with torch.no_grad(): #其实有了model_eval 也可以不用no_grad,用了就能速度更快一点
        for index, (numpairs,attr,label) in bar:
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
    return a

    # dd = pd.DataFrame(np.array([[i[0:3]] for i in results]).squeeze(1), columns=["Mre", "Ma", "Me"])
    # sns.lineplot(data=dd)
    # plt.show()

def Mre_Ma_Mr(indices,label):
    Mre=(Levenshtein.distance(indices,label))/( len(indices) if len(indices)>len(label) else len(label))
    Ma=sum([1 if i in indices else 0 for i in indices[:len(label)]])/ len(indices) #预测标签的长度
    Mr=sum([1 if i in indices else 0 for i in indices[:len(label)]]) / len(label)  #真实值的长度
    return Mre,Ma,Mr


# def numpy_accuracy(results,labels):
#     '''
#                     使用numpy 将预测值 和 真实值 进行比较，并返回两个矩阵中对应元素是否相等的布尔值
#                     布尔值可以直接求和
#                 '''
#     total_correct = sum(np.array(results).T == np.array(labels).T)  # 得到预测正确的个数
#     acc = np.mean(total_correct / (config.day_walk_max + 1))
#     print('模型的预测准确率为:{:.3f}'.format(acc))
def space_loss(topi, targeti):
    # predict_xy = .index2word[topi.squeeze().detach().item()]
    # predict_xy = predict_xy.split('P')
    # target_xy = output_lang.index2word[targeti.squeeze().detach().item()]
    # target_xy = target_xy.split('P')
    # delta_x = np.square(float(predict_xy[0]) - float(target_xy[0]))
    # delta_y = np.square(float(predict_xy[1]) - float(target_xy[1]))
    # return np.sqrt(delta_x + delta_y)
    pass

def _prepar_seq(seq):#对结果进行基础的处理，共后续转化为文字使用
    seq=seq.view(-1)
    if  seq[0].item() == Num_sequence.SOS:
        seq=seq[1:]
    #裁尾巴
    seq = [i.item() for i in seq if i.item()>3]
    return seq
if __name__ == '__main__':
    seq2seq = Seq2Seq().to(config.device) # 实例化模型

    # seq2seq.load_state_dict(torch.load(config.model_save_path,map_location='cpu'))  # 加载模型
    seq2seq.load_state_dict(torch.load(config.model_save_path))  # 加载模型
    # print("基于greedy_eval", 10 * "*")
    # eval()
    print("基于beam_search的eval",10*"*")

    print(eval_beam_search(seq2seq,3))