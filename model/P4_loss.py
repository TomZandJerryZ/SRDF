import torch.nn.functional as F
import torch
import torch.nn as nn


# 用于计算P4层的损失
#

def gauss_reg_loss(pred,target,mask):
    # print(pred.shape, target.shape)
    batch_size = target.shape[0]
    c = target.shape[-1]
    preds_reshape=[]
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos = mask.sum().clamp_(min=1).float()#[batch_size,]

    pred=pred.permute(0,2,3,1)
    # print(pred.shape, target.shape)
    assert pred.shape==target.shape#[batch_size,sum(_h*_w),4]
    loss=[]
    for batch_index in range(batch_size):

        pred_pos=pred[batch_index][mask[batch_index]]

        target_pos=target[batch_index][mask[batch_index]]

        # assert len(pred_pos.shape)==2

        loss.append(F.smooth_l1_loss(pred_pos,target_pos,reduction='sum').view(1))

    return torch.cat(loss,dim=0)/num_pos#[batch_size,]


def GaussLoss( pred, target):
    # print(pred.shape,target.shape)
    pred = pred.float()
    pred = pred.sigmoid()
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)

    loss = 0

    pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, 2) * pos_inds

    neg_loss = torch.log(1 - pred + 1e-6) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    # print(loss)
    return loss


from torch.cuda.amp import autocast


def Gauss_attLoss(pred,target):
    batch_size = target.shape[0]
    loss = []
    for batch_index in range(batch_size):
        pred_pos=pred[batch_index]#[num_pos_b,]
        # print(pred_pos.shape)
        target_pos=target[batch_index]#[num_pos_b,]
        with autocast(enabled=False):
            L = nn.functional.binary_cross_entropy(input=pred_pos, target=target_pos, reduction='mean')
        loss.append(L.view(1))
    return torch.cat(loss,dim=0)#[batch_size,]

