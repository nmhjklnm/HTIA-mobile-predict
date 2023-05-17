
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import config



class NumDataset(Dataset):
    def __init__(self):
        '''
        @param train: 默认为True，表示生成测试集
        '''
        lines = open(config.txtpath).read().strip().split('\n')
        # 行分割并标准化
        pairs = [[s for s in l.split('#')] for l in lines]
        self.numpair = []       # 用来记录每天的轨迹
        self.attr = []          # 用来记录每天的属性
        self.label = []

        for week_rank in range(len(pairs)):
            self.label += [pairs[week_rank][7].split(',')[:-1]]
            for day in range(7):
                self.numpair += [pairs[week_rank][day].split('Attr')[0].split(',')[:-1]]
                self.attr.append(pairs[week_rank][day].split('Attr')[1].split(',')[1:])

    def __getitem__(self, item):
        '''
            传入模型的应该是字符串，而不是数字
            list将字符串转换为列表
        '''
        numpair=self.numpair[7*item:item*7+7]
        attr   =self.attr[7 * item:item * 7+7]
        label=self.label[item]
        return numpair,attr,label

    def __len__(self):
        return len(self.label)

def collate_fn_train(batch):
    '''
        zip(*batch) 对每一个batch进行解包
        [(input1, label1), (input2, label2)] --> [(input1, input2), (label1, label2)]

        batch:[(input1, lable1), ...]
        使用编码器编码的时候 需要对句子进行 从长到短的排序，故可以在collate_fn中提前将这一部分完成
        # 根据batch的第三个值的大小进行排序，reverse=True取反表示从大到小
        sorted(batch, key=lambda x:x[2], reverse=True)
    '''
    # batch = sorted(batch, key=lambda x:x[2], reverse=True)
    numpair,attr,label = zip(*batch)
    numpairs=[] #存放batch轨迹
    for single_input_sample in numpair:
        """
        just_for_padding用来填充周序列
         sum(a,[]),用来展开列表
         # for single_input_sample in numpair:
        #     _=[config.num_sequence.transform(day_traj) for day_traj in single_input_sample]
        #     numpairs.append(_)
        # 对每一个输入进行处理，入一周轨迹
        """
        _= [config.num_sequence.transform(day_traj,max_len=config.day_walk_max) for day_traj in single_input_sample]
        numpairs.append(_)
    numpairs = torch.LongTensor(numpairs).view(config.batch_size, 7 * config.day_walk_max)
    label = [config.num_sequence.transform(i, max_len=config.day_walk_max,add_eos=True) for i in label]
    attr=torch.LongTensor(np.array(attr)[:,:,1:3].astype(int))
    return numpairs,attr,torch.LongTensor(label)

def collate_fn_test(batch):
    '''
        zip(*batch) 对每一个batch进行解包
        [(input1, label1), (input2, label2)] --> [(input1, input2), (label1, label2)]

        batch:[(input1, lable1), ...]
        使用编码器编码的时候 需要对句子进行 从长到短的排序，故可以在collate_fn中提前将这一部分完成
        # 根据batch的第三个值的大小进行排序，reverse=True取反表示从大到小
        sorted(batch, key=lambda x:x[2], reverse=True)
    '''
    # batch = sorted(batch, key=lambda x:x[2], reverse=True)
    numpair,attr,label = zip(*batch)
    numpairs=[] #存放batch轨迹
    for single_input_sample in numpair:
        """
        just_for_padding用来填充周序列
         sum(a,[]),用来展开列表
         # for single_input_sample in numpair:
        #     _=[config.num_sequence.transform(day_traj) for day_traj in single_input_sample]
        #     numpairs.append(_)
        # 对每一个输入进行处理，入一周轨迹
        """
        _= [config.num_sequence.transform(day_traj,max_len=config.day_walk_max) for day_traj in single_input_sample]
        numpairs.append(_)
    numpairs = torch.LongTensor(numpairs).view(config.test_batch_size, 7 * config.day_walk_max)
    label = [config.num_sequence.transform(i, max_len=config.day_walk_max,add_eos=True) for i in label]
    attr=torch.LongTensor(np.array(attr)[:,:,1:3].astype(int))
    return numpairs,attr,torch.LongTensor(label)


# def collate_fn_test(batch):
#     '''
#         zip(*batch) 对每一个batch进行解包
#         [(input1, label1), (input2, label2)] --> [(input1, input2), (label1, label2)]
#
#         batch:[(input1, lable1), ...]
#         使用编码器编码的时候 需要对句子进行 从长到短的排序，故可以在collate_fn中提前将这一部分完成
#         # 根据batch的第三个值的大小进行排序，reverse=True取反表示从大到小
#         sorted(batch, key=lambda x:x[2], reverse=True)
#     '''
#     # batch = sorted(batch, key=lambda x:x[2], reverse=True)
#     numpair,attr,label = zip(*batch)
#     numpairs=[] #存放batch轨迹
#     for single_input_sample in numpair:
#         """
#         just_for_padding用来填充周序列
#          sum(a,[]),用来展开列表
#          # for single_input_sample in numpair:
#         #     _=[config.num_sequence.transform(day_traj) for day_traj in single_input_sample]
#         #     numpairs.append(_)
#         # 对每一个输入进行处理，入一周轨迹
#         """
#         _= [config.num_sequence.transform(day_traj,max_len=config.day_walk_max) for day_traj in single_input_sample]
#         numpairs.append(_)
#     numpairs = torch.LongTensor(numpairs).view(config.test_batch_size, 7 * config.day_walk_max)
#     label = [config.num_sequence.transform(i) for i in label]
#     attr=torch.LongTensor(np.array(attr)[:,:,1:3].astype(int))
#     return numpairs,attr,label

train_size = int(config.train_coll * len(NumDataset()))
train_dataset, test_dataset = torch.utils.data.random_split(NumDataset(), [train_size, len(NumDataset()) - train_size],generator=torch.manual_seed(config.manu_seed))
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn_train, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch_size,shuffle=True, collate_fn=collate_fn_test, drop_last=True)

if __name__ == '__main__':
    numpair,attr,label=NumDataset()[0]
    for i,s,v in train_dataloader:
        print(len(s))
        break