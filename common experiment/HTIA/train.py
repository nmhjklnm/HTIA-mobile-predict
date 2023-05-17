import torch
from torch.optim import Adam
import torch.nn.functional as F
from seq2seq.seq2seq_model import Seq2Seq
from data.dataset import train_dataloader
from eval import eval_beam_search
import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

seq2seq = Seq2Seq().to(config.device)
optimizer = Adam(seq2seq.parameters(), lr=config.learning_rate)
writer = SummaryWriter()
def train(epoch):

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ascii=True)
    for index, (numpairs,attr,label) in bar:

        seq2seq.train()
        numpairs = numpairs.to(config.device)
        attr= attr.to(config.device)
        label= label.to(config.device)

        optimizer.zero_grad()
        decoder_outputs, _ = seq2seq(numpairs,attr,label)
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(0) * decoder_outputs.size(1), -1)
        label = label.view(-1)
        loss = F.nll_loss(decoder_outputs, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seq2seq.parameters(),0.01)
        optimizer.step()
        bar.set_description('train epoch:{}\tindex:{}\tloss:{:.3f}\t'.format(epoch, index, loss.item()))

        tenserbroad(loss.item() ,epoch)
        if epoch==99 and index==len(bar)-1:
            torch.save(seq2seq.state_dict(), config.model_save_path)
            torch.save(optimizer.state_dict(), config.optimizer_save_path)


def tenserbroad(loss ,epoch):
    writer.add_scalar("loss", loss, global_step=epoch)



if __name__ == '__main__':
    for i in range(100):
        train(i)
    k3val = eval_beam_search(seq2seq, 3)
    writer.add_hparams({'data': config.data
                           , 'manu_seed': config.manu_seed
                           , 'hidden_size': config.hidden_size
                           , 'bsize': config.batch_size
                           , 'lr': config.learning_rate
                        },
                        {'Mre3': k3val[0]
                            , 'Ma3': k3val[1]
                            , 'Mr3': k3val[2]
                        },
                      )
    writer.close()
