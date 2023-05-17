


import config


class Num_sequence():

    PAD_TAG = 'PAD'
    UNK_TAG = 'UNK'
    SOS_TAG = 'SOS'
    EOS_TAG = 'EOS'

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
        print("create dict",10*"*")
        self.generate_dict()
        print("nodelen：",len(self.dict)-4)
        print("samplelen",self.sample_len)
        print("max_traj",self.sample_day_walk)

    def transform(self, sentence, max_len=None, add_eos=False):

        if max_len==None:
            self._fit_dict(sentence)
            result = [self.dict.get(i, self.UNK) for i in sentence]+[self.EOS]*add_eos
        else:
            self._fit_dict(sentence)

            sentence = sentence[:max_len] +[self.EOS_TAG]*add_eos  \
                if len(sentence) >= max_len else \
                sentence +[self.EOS_TAG]*add_eos+ [self.PAD_TAG] * (max_len - len(sentence))

            result = [self.dict.get(i, self.UNK) for i in sentence]
        return result

    def inverse_transform(self, indices):

        return [self.inverse_dict.get(i, self.UNK_TAG) for i in indices]

    def _fit_dict(self,sentence):

        for traj_str in sentence:
            if traj_str not in self.dict:
                self.dict[traj_str] = len(self.dict)
            self.count[traj_str]=self.count.get(traj_str, 0) + 1
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
    def generate_dict(self):
        lines = open(config.txtpath,encoding='utf-8').read().strip().split('\n')
        pairs = [[s for s in l.split('#')] for l in lines]
        self.sample_len = len(pairs)
        numpair,label = [],[]
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
    pass
