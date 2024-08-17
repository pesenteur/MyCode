import torch
import torch.nn as nn

def inner_product(mat_1, mat_2):
    n, m = mat_1.shape 
    mat_expand = torch.unsqueeze(mat_2, 0)  
    mat_expand = mat_expand.expand(n, n, m)  
    mat_expand = mat_expand.permute(1, 0, 2)  
    inner_prod = torch.mul(mat_expand, mat_1)  
    inner_prod = torch.sum(inner_prod, axis=-1)  
    return inner_prod

def mobility_loss(s_embeddings, t_embeddings, mob):
    inner_prod = inner_product(s_embeddings, t_embeddings)
    softmax1 = nn.Softmax(dim=-1)
    phat = softmax1(inner_prod)
    loss = torch.sum(-torch.mul(mob, torch.log(phat+0.0001)))
    inner_prod = inner_product(t_embeddings, s_embeddings)
    softmax2 = nn.Softmax(dim=-1)
    phat = softmax2(inner_prod)
    loss += torch.sum(-torch.mul(torch.transpose(mob, 0, 1), torch.log(phat+0.0001)))
    return loss

class MobilityLoss(nn.Module):
    def __init__(self):
        super(MobilityLoss, self).__init__()

    def forward(self, out1, out2, label):
        mob_loss = mobility_loss(out1, out2, label)
        return mob_loss
