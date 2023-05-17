import config
import torch
def embedding_concat(embeded_traj,embeded_attr):

    temp = torch.zeros([embeded_traj.size(0), config.day_walk_max*7,config.attr_embedding_dim * 2],device=config.device)

    for i in range(7):
        temp[:,i*20:20*i+20,:]=embeded_attr[:,i:i+1,:].repeat(1,20,1)
    return torch.concat([embeded_traj,temp],dim=2)