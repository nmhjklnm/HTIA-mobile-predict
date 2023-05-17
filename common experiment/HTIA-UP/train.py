

import torch
from torch.optim import Adam
import torch.nn.functional as F
from seq2seq.seq2seq_model import Seq2Seq
from data.dataset import train_dataloader
from eval import eval, eval_beam_search,eval_in_train
import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


seq2seq = Seq2Seq().to(config.device)  # 将模型分配给指定设备
# optimizer = Adam(seq2seq.parameters(), lr=1e-3,weight_decay=0.00005)
optimizer = Adam(seq2seq.parameters(), lr=config.learning_rate)
writer = SummaryWriter()
def train(epoch):

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=True)
    for index, (numpairs,attr,label) in bar:

        seq2seq.train()
        # 将训练数据分配给指定设备
        numpairs = numpairs.to(config.device)
        attr= attr.to(config.device)
        label= label.to(config.device)

        optimizer.zero_grad()   # 梯度置0
        decoder_outputs, _ = seq2seq(numpairs,attr,label)

        # if epoch==0 and index==0:
        #     writer.add_graph(Seq2Seq().to("cpu"),[numpairs.to("cpu"),attr.to("cpu"),label.to("cpu")])

        acc = eval(decoder_outputs, label) if epoch>80 else [0,0,0]  #对后二十个epoch进行测试,训练集的测试
        # acc=[0,0,0]
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(0) * decoder_outputs.size(1), -1)  # [batch_size*max_len, vacab_size]
        label = label.view(-1)  # [batch_size*max_len]

        loss = F.nll_loss(decoder_outputs, label)  # 计算loss
        loss.backward()   # 反向传播
        torch.nn.utils.clip_grad_norm_(seq2seq.parameters(),0.01) #进行梯度裁剪，防止梯度过大
        optimizer.step()  # 参数更新

        bar.set_description('train epoch:{}\tindex:{}\tloss:{:.3f}\tacc:{}'.format(epoch, index, loss.item(),acc))

        val_acc=eval_in_train(seq2seq,3) if epoch>99 and index==len(bar)-1 else [0,0,0] #这个是测试集的评估  and epoch %2==0 and index==len(bar)-1

        tenserbroad(loss.item(), acc,val_acc ,epoch)  #用途如同函数名

        if epoch==99 and index==len(bar)-1:#虽然有6个epoch，但有0到5
            torch.save(seq2seq.state_dict(), config.model_save_path)
            torch.save(optimizer.state_dict(), config.optimizer_save_path)
def tenserbroad(loss, acc,val_acc ,epoch):
    writer.add_scalar("loss", loss, global_step=epoch)
    name=("train","val")
    Mre,Ma,Mr=[dict(zip(name,i)) for i in zip(acc,val_acc)]
    writer.add_scalars("Mre", Mre,  global_step=epoch)
    writer.add_scalars("Ma", Ma , global_step=epoch)
    writer.add_scalars("Mr", Mr , global_step=epoch)



if __name__ == '__main__':

    for i in range(100):
        train(i)

    print("beam search topk3")
    k3val=eval_beam_search(seq2seq,3)
    print("beam search topk1")
    k1val=eval_beam_search(seq2seq,1)
    writer.add_hparams({'data': config.data
                           , 'manu_seed': config.manu_seed
                           , 'hidden_size': config.hidden_size
                           , 'bsize': config.batch_size
                           , 'lr': config.learning_rate
                        },
                       {'Mre3': k3val[0]
                           ,'Ma3' : k3val[1]
                           ,'Mr3' : k3val[2]
                           ,'Mre1': k1val[0]
                           , 'Ma1': k1val[1]
                           , 'Mr1': k1val[2]})
    writer.close()
