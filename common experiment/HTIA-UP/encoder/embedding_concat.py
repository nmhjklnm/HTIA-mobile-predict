import config
import torch
def embedding_concat(embeded_traj,embeded_attr):
    """
    思路是扩充attr维度到traj一致的程度
    @param embeded_traj:
    @param embeded_attr:
    @return:
    """
    #temp 初始化，用来存放结果
    temp = torch.zeros([embeded_traj.size(0), config.day_walk_max*7,config.attr_embedding_dim * 2],device=config.device)
    #写个for循环用来单独改造embedded_attr ,目标是改造成和embeded同一维度
    for i in range(7):
        temp[:,i*20:20*i+20,:]=embeded_attr[:,i:i+1,:].repeat(1,20,1)
    return torch.concat([embeded_traj,temp],dim=2)