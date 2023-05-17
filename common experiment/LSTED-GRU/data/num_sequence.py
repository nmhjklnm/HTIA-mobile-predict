


import config


class Num_sequence():
    '''
    自然语言处理常用标识符:
        <UNK>: 低频词或未在词表中的词
        <PAD>: 补全字符
        <GO>/<SOS>: 句子起始标识符
        <EOS>: 句子结束标识符
        [SEP]：两个句子之间的分隔符
        [MASK]：填充被掩盖掉的字符
    '''
    PAD_TAG = 'PAD'
    UNK_TAG = 'UNK'
    SOS_TAG = 'SOS'  # 句子开始符号
    EOS_TAG = 'EOS'  # 结束符

    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {
            self.PAD_TAG: self.PAD,
            self.UNK_TAG: self.UNK,
            self.SOS_TAG: self.SOS,
            self.EOS_TAG: self.EOS,
        }
        self.count={}
        print("开始构建字典",10*"*")
        self.generate_dict()
        print("节点数：",len(self.dict)-4)
        print("样本数",self.sample_len)
        print("当天最长轨迹",self.sample_day_walk)

    def transform(self, sentence, max_len=None, add_eos=False):
        """
        两种输出方式，如果不设置None就不进行裁剪和填补
        @param sentence:
        @param max_len:
        @param add_eos:
        @return:
        """
        if max_len==None:
            self._fit_dict(sentence)
            result = [self.dict.get(i, self.UNK) for i in sentence]+[self.EOS]*add_eos
        else:
            self._fit_dict(sentence)
            '''
            把sentence转化为数字序列
               sentence: ['1', '2', '4', ...] --> [1, 2, 4, ...]
            句子大于最大长度，需要进行裁剪
                    这里减1的目的是使得句子的长度保持一致
                    以 句子长度=11， 最大长度=10 为例:
                        因为 EOS也相当一个字符，使用代码sentence = sentence[:max_len]对句子进行切割后，再加上EOS
                        那么 句子的长度就会超过 最大限制10
    
                        因此 使用 max_len-1
                        做法就是提前留出一个位置给 EOS
                        这样就能保证句子的最大长度总是10
                        如果句子小于最大长度， 则需要在句子后面补充PAD占位符
            '''
            sentence = sentence[:max_len] +[self.EOS_TAG]*add_eos  \
                if len(sentence) >= max_len else \
                sentence +[self.EOS_TAG]*add_eos+ [self.PAD_TAG] * (max_len - len(sentence))
            # 如果词语未在字典中出现则用UNK替代
            result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def inverse_transform(self, indices):
        '''把序列转回字符串'''
        return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]

    def _fit_dict(self,sentence):
        """
        @param sentence:"weektraj_list_str"
        @return:        "用来对self.dict技术及编码",自动计入这个类的dict中了
        """
        for traj_str in sentence:
            if traj_str not in self.dict: #不重复编码，只记录第一次
                self.dict[traj_str] = len(self.dict) #轨迹编码字典
            self.count[traj_str]=self.count.get(traj_str, 0) + 1 #轨迹计数字典
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))  # 快速得到轨迹解码字典
    def generate_dict(self):
        lines = open(config.txtpath,encoding='utf-8').read().strip().split('\n')
        # 行分割并标准化
        pairs = [[s for s in l.split('#')] for l in lines]
        self.sample_len = len(pairs)
        numpair,label = [],[]  # 用来记录每天的轨迹    所要预测的轨迹
        for week_rank in range(len(pairs)):
            label += [pairs[week_rank][7].split(',')[:-1]]
            for day in range(7):
                numpair += [pairs[week_rank][day].split('Attr')[0].split(',')[:-1]]
        self._fit_dict(sum(numpair,[]))
        self._fit_dict(sum(label,[]))
        self.sample_day_walk=max(max([len(i) for i in numpair]),max([len(i) for i in label]))


    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    numseq = Num_sequence()
    print(numseq.dict)
    print(numseq.inverse_dict)
    print(numseq.count)