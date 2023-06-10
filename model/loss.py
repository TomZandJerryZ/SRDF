import torch
import torch.nn as nn
from .config import DefaultConfig
import torch.nn.functional as F
from dataloader.bbox import quad_2_rbox,rbox_2_quad,sort_corners
from torch.cuda.amp import autocast
from .genTarget_np import np_gen_level_targets,batch_gen_target
from .gauss_p2 import batch_gen_gausstarget
from dataloader.BoxCoder import *
import torch

from .P4_loss import GaussLoss,gauss_reg_loss,Gauss_attLoss
def coords_fmap2orig(feature,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
    h,w=feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords

class GenTargets(nn.Module):
    def __init__(self,strides,limit_range,gauss_range):
        super().__init__()
        self.strides=strides
        self.limit_range=limit_range
        self.gauss_range = gauss_range
        assert len(strides)==len(limit_range)

    def forward(self,inputs):

        cls_logits, cnt_logits, reg_preds, theta_cls_lo,theta_reg_lo = inputs[0]

        gt_boxes=inputs[1]
        classes=inputs[2]

        cls_targets_all_level=[]
        cnt_targets_all_level=[]
        reg_targets_all_level=[]
        tc_targets_all_level=[]
        tr_targets_all_level=[]
        target = {}
        assert len(self.strides)==len(cls_logits)
        # P2 gauss层
        # P2_level_out = [cls_logits[0], cnt_logits[0], reg_preds[0]]
        # P2cls_targets, P2cnt_targets ,P2reg_targets = batch_gen_gausstarget(batch_gt=gt_boxes, batch_cls=classes, level_out=P2_level_out,
        #                                      stride=self.strides[0], limit_range=self.limit_range[0])
        # # for i in P2_level_out:
        # #     print(i.shape)
        # target['P2'] = [P2cls_targets, P2cnt_targets ,P2reg_targets]
        # # 在这里进行P2之前层数的target
        for level in range(0,len(cls_logits)):
        # for level in range(0, len(cls_logits)):
            # print(cnt_logits[level].shape) # b 1 h w
            level_out=[cls_logits[level],cnt_logits[level],reg_preds[level]]


            level_targets = batch_gen_target(batch_gt=gt_boxes, batch_cls=classes, level_out=level_out,
                                             stride=self.strides[level], limit_range=self.limit_range[level],gauss_range = self.gauss_range[level])

            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])
            tc_targets_all_level.append(level_targets[3])
            tr_targets_all_level.append(level_targets[4])

        target['P5_7'] = [torch.cat(cls_targets_all_level,dim=1),torch.cat(cnt_targets_all_level,dim=1),torch.cat(reg_targets_all_level,dim=1),torch.cat(tc_targets_all_level,dim=1),torch.cat(tr_targets_all_level,dim=1)]
        return target

    # def in_gbox(self,gt_boxess,coords):
    #     # gt_boxes [batch_size,m,8]
    #     if gt_boxess.shape[-1] == 4:
    #         b,m = gt_boxess.shape[0], gt_boxess.shape[1]
    #         gt_boxes = torch.zeros((b,m,8)).to(device=gt_boxess.device)
    #         gt_boxes[..., 0] = gt_boxess[..., 0]
    #         gt_boxes[..., 1] = gt_boxess[..., 1]
    #         gt_boxes[..., 2] = gt_boxess[..., 2]
    #         gt_boxes[..., 3] = gt_boxess[..., 1]
    #         gt_boxes[..., 4] = gt_boxess[..., 2]
    #         gt_boxes[..., 5] = gt_boxess[..., 3]
    #         gt_boxes[..., 6] = gt_boxess[..., 0]
    #         gt_boxes[..., 7] = gt_boxess[..., 3]
    #
    #     else:
    #         gt_boxes = gt_boxess
    #
    #
    #
    #     gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2] + gt_boxes[..., 4] +gt_boxes[..., 6]) / 4
    #     gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]+ gt_boxes[..., 5] +gt_boxes[..., 7]) / 4
    #     gt_center = torch.cat([gt_center_x.unsqueeze(-1),gt_center_y.unsqueeze(-1)],dim=-1)
    #     # print(gt_boxes[...,0:2].shape,gt_center.shape)
    #     # (x,y) - (x_c,y_c) = (x',y')
    #     # (x,y) - alh*(x',y') = (new_x,new_y)
    #     alpha = 0.2
    #     gt_boxes[...,0:2] = gt_boxes[...,0:2] - (gt_boxes[...,0:2]-gt_center)*alpha
    #     gt_boxes[..., 2:4] = gt_boxes[..., 2:4] - (gt_boxes[..., 2:4] - gt_center) * alpha
    #     gt_boxes[..., 4:6] = gt_boxes[..., 4:6] - (gt_boxes[..., 4:6] - gt_center) * alpha
    #     gt_boxes[..., 6:] = gt_boxes[..., 6:] - (gt_boxes[..., 6:] - gt_center) * alpha
    #
    #     biu = torch.zeros_like(coords).to(device=gt_boxess.device)
    #
    #     # print(gt_boxes[0,0,:],gt_boxess[0,0,:])
    #     AB = (gt_boxes[...,2:4] - gt_boxes[...,0:2])[:,None,:] - biu[None,:,None]
    #     BC = (gt_boxes[...,4:6] - gt_boxes[...,2:4])[:,None,:] - biu[None,:,None]
    #     CD = (gt_boxes[...,6:] - gt_boxes[...,4:6])[:,None,:] - biu[None,:,None]
    #     DA = (gt_boxes[...,0:2] - gt_boxes[...,6:])[:,None,:] - biu[None,:,None]
    #
    #     # print(AB)
    #     AM = coords[None,:,None] - gt_boxes[...,0:2][:,None,:]
    #     BM = coords[None,:,None] - gt_boxes[...,2:4][:,None,:]
    #     CM = coords[None,:,None] - gt_boxes[...,4:6][:,None,:]
    #     DM = coords[None,:,None] - gt_boxes[...,6:][:,None,:]
    #
    #     #[b,h*w,m,2]
    #     AB_AM = AB[:,:,:,0]*AM[:,:,:,0] + AB[:,:,:,1]*AM[:,:,:,1]
    #     BC_BM = BC[:,:,:,0]*BM[:,:,:,0] + BC[:,:,:,1]*BM[:,:,:,1]
    #     CD_CM = CD[:,:,:,0]*CM[:,:,:,0] + CD[:,:,:,1]*CM[:,:,:,1]
    #     DA_DM = DA[:,:,:,0]*DM[:,:,:,0] + DA[:,:,:,1]*DM[:,:,:,1]
    #     # print(AB_AM.shape)
    #
    #     mask_in_gt = (AB_AM > 0) & (BC_BM >0)& (CD_CM >0)& (DA_DM >0)
    #     return mask_in_gt,gt_center
    #
    # def five_to_eight(self,gt_boxes):
    #     '''
    #     gt_boxes[batch_size, m, 5]
    #     xyxya
    #     '''
    #     # print(gt_boxes.shape)
    #     eight = []
    #     for i in range(0,gt_boxes.shape[0]):
    #         dets = gt_boxes[i].cpu().numpy()
    #         res = sort_corners(rbox_2_quad(dets))
    #         eight.append( torch.from_numpy(res).cuda())
    #
    #     return torch.stack(eight,dim=0)
    # def _gen_level_targets(self,out,gt_boxes,classes,stride,limit_range,sample_radiu_ratio=1.5):
    #     '''
    #     Args
    #     out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]
    #     gt_boxes [batch_size,m,4]
    #     classes [batch_size,m]
    #     stride int
    #     limit_range list [min,max]
    #     Returns
    #     cls_targets,cnt_targets,reg_targets
    #     '''
    #     cls_logits,cnt_logits,reg_preds,theta_logits ,theta_reg=out
    #     # cls_logits, cnt_logits, reg_preds,  theta_reg = out
    #     batch_size=cls_logits.shape[0]
    #     class_num=cls_logits.shape[1]
    #     m=gt_boxes.shape[1]
    #
    #     cls_logits=cls_logits.permute(0,2,3,1) #[batch_size,h,w,class_num]
    #     coords=coords_fmap2orig(cls_logits,stride).to(device=gt_boxes.device)#[h*w,2]
    #
    #     cls_logits=cls_logits.reshape((batch_size,-1,class_num))#[batch_size,h*w,class_num]
    #     cnt_logits=cnt_logits.permute(0,2,3,1)
    #     cnt_logits=cnt_logits.reshape((batch_size,-1,1))
    #     reg_preds=reg_preds.permute(0,2,3,1)
    #     reg_preds=reg_preds.reshape((batch_size,-1,5))
    #
    #     h_mul_w=cls_logits.shape[1]
    #     # classes = gt_boxes[:,-1]
    #     # print(gt_boxes.shape)
    #     x=coords[:,0]
    #     y=coords[:,1]
    #     eight_box = self.five_to_eight(gt_boxes)
    #     mig, _ = self.in_gbox(eight_box, coords)
    #     biu = torch.zeros_like(x).to(device=gt_boxes.device)
    #     # gt_center_x = (gt_boxes[..., 2]+gt_boxes[..., 0])/2
    #     # gt_center_y = (gt_boxes[..., 3]+gt_boxes[..., 1])/2
    #     # w = (gt_boxes[..., 2]-gt_boxes[..., 0])
    #     # # print(w.shape)
    #     # h = (gt_boxes[..., 3]-gt_boxes[..., 1])
    #     #
    #     #
    #     # biu_w = w[:,None,:] - biu[None,:,None]#[batch_size,h*w,m]
    #     # # print(biu_w.shape)
    #     # biu_h = h[:,None,:] - biu[None,:,None]
    #     # cx_off = gt_center_x[:,None,:] - x[None,:,None]
    #     # cy_off = gt_center_y[:,None,:] - y[None,:,None]
    #     #
    #     # biu_off = torch.stack([biu_w,biu_h,cx_off,cy_off],dim=-1)
    #
    #     import numpy as np
    #     # theta = gt_boxes[...,-1]#[batch_size,m]
    #     # import numpy as np
    #     # biu_off = torch.stack([biu_x1, biu_y1, biu_x2, biu_y2, biu_x3, biu_y3, biu_x4, biu_y4], dim=-1)
    #     theta = gt_boxes[..., -1]  # [batch_size,m]
    #     ex_thetas = torch.zeros_like(theta).to(device=gt_boxes.device)
    #
    #     targets_dt = 15 * (torch.tan((45 - theta) / 180.0 * np.pi) - torch.tan(ex_thetas / 180.0 * np.pi))
    #     targets_dt = targets_dt[:, None, :] - biu[None, :, None]
    #     # print(theta)
    #     # dt = theta
    #     # dt = (torch.tan(theta / 180.0 * np.pi) - torch.tan(ex_thetas / 180.0 * np.pi)) *15
    #
    #     # pre_t = torch.atan(torch.tan(ex_thetas / 180.0 * np.pi) + dt) / np.pi * 180.0
    #     #
    #     # print(theta[0][0])
    #     # print(pre_t[0][0])
    #     theta_c = theta // 10
    #     dt = theta - theta_c
    #     # theta_r = theta_r[:,None,:] - biu[None,:,None]
    #     theta_r = dt[:, None, :] - biu[None, :, None]
    #     # areas=(biu_w)*(biu_h)#[batch_size,h*w,m]
    #
    #     # off_min=torch.min(ltrb_off,dim=-1)[0]#[batch_size,h*w,m]
    #     # off_max=torch.max(torch.stack([biu_w,biu_h],dim=-1),dim=-1)[0]#[batch_size,h*w,m]
    #     # 这里直接用wh来表示目标是不是不合适 获取使用max x -minx maxy -miny来表示比较合适
    #
    #     l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
    #     '''###############'''
    #     t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
    #     r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
    #     b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
    #     ltrb_off = torch.stack([l_off, t_off, r_off, b_off,targets_dt], dim=-1)  # [batch_size,h*w,m,4]
    #
    #     areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])  # [batch_size,h*w,m]
    #
    #     off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
    #     off_max = torch.max(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
    #
    #     mask_in_gtboxes = off_min > 0  # 布尔变量 证明mask是否在groundtruth里面
    #
    #     # mask_in_gtboxes=off_min>0
    #     # mask_in_level=(off_max>limit_range[0])&(off_max<=limit_range[1])
    #     mask_in_level = (areas > limit_range[0]) & (areas <= limit_range[1])
    #
    #     radiu = stride * sample_radiu_ratio
    #     gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
    #     gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
    #     c_l_off = x[None, :, None] - gt_center_x[:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
    #     c_t_off = y[None, :, None] - gt_center_y[:, None, :]
    #     # c_r_off = gt_center_x[:, None, :] - x[None, :, None]
    #     # c_b_off = gt_center_y[:, None, :] - y[None, :, None]
    #     # c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)  # [batch_size,h*w,m,4]
    #     # c_off_max = torch.max(c_ltrb_off, dim=-1)[
    #     #     0]  # 对于每个输出层的正样本区域，遍历每个point位置，计算其max（left/top/right/bottom）的target是否在指定范围内
    #     # # 不在范围内的认为是背景区域
    #     circle_off = torch.sqrt(c_l_off**2+c_t_off**2)
    #     # mask_center = c_off_max < radiu
    #     mask_center = circle_off < (2* stride)
    #
    #
    #     # mask_pos=mask_in_gtboxes&mask_in_level&mask_center#[batch_size,h*w,m]
    #     mask_pos =  mask_in_level & mig & mask_center  # [batch_size,h*w,m]
    #
    #     m2 = mask_in_level & mig
    #     # areas[~m2] = 999999
    #     # cnt = torch.min(areas,dim=-1)[1]
    #     areas[~mask_pos]=999999
    #     areas_min_ind=torch.min(areas,dim=-1)[1]#[batch_size,h*w]
    #
    #     reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1),
    #                                                                               1)]  # [batch_size*h*w,4]
    #     reg_targets=torch.reshape(reg_targets,(batch_size,-1,5))#[batch_size,h*w,4]
    #     reg_targets[:,:,0:4] = reg_targets[:,:,0:4] / stride
    #     # classes
    #     if len(classes.size()) == 3:
    #         classes = classes.squeeze(-1)
    #     # print(classes.shape)
    #     classes=torch.broadcast_tensors(classes[:,None,:],areas.long())[0]#[batch_size,h*w,m]
    #     # print(classes[m2][classes[m2]==-1])
    #     cls_targets=classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
    #     cls_targets=torch.reshape(cls_targets,(batch_size,-1,1))#[batch_size,h*w,1]
    #     # cnt_targets = classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,cnt.unsqueeze(dim=-1),1)]
    #     # cnt_targets = torch.reshape(cnt_targets, (batch_size, -1, 1))
    #
    #     theta_c = torch.broadcast_tensors(theta_c[:, None, :], areas.long())[0]  # [batch_size,h*w,m]
    #     theta_c_targets = theta_c[
    #         torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
    #     theta_c_targets = torch.reshape(theta_c_targets, (batch_size, -1, 1))  # [batch_size,h*w,1]
    #
    #     theta_r_targets = theta_r[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1),
    #                                                                              1)]  # [batch_size*h*w,4]
    #     theta_r_targets = torch.reshape(theta_r_targets, (batch_size, -1, 1))  # [batch_size,h*w,4]
    #
    #
    #     # left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])#[batch_size,h*w]
    #     # left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
    #     # top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
    #     # top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
    #     # left_right_min = torch.min(reg_targets1[..., 0], reg_targets1[..., 2])  # [batch_size,h*w]
    #     # left_right_max = torch.max(reg_targets1[..., 0], reg_targets1[..., 2])
    #     # top_bottom_min = torch.min(reg_targets1[..., 1], reg_targets1[..., 3])
    #     # top_bottom_max = torch.max(reg_targets1[..., 1], reg_targets1[..., 3])
    #     # cnt_targets=((left_right_min*top_bottom_min)/(left_right_max*top_bottom_max+1e-10)).sqrt().unsqueeze(dim=-1)#[batch_size,h*w,1]
    #     cnt_targets = torch.zeros_like(cls_targets)
    #     assert reg_targets.shape==(batch_size,h_mul_w,5)
    #     assert cls_targets.shape==(batch_size,h_mul_w,1)
    #     assert cnt_targets.shape==(batch_size,h_mul_w,1)
    #
    #     #process neg coords
    #     mask_pos_2 = mask_pos.long().sum(dim=-1)#[batch_size,h*w]
    #     # mask_cnt = m2.long().sum(dim=-1)>=1
    #     mask_pos_2=mask_pos_2>=1
    #     assert mask_pos_2.shape==(batch_size,h_mul_w)
    #     cls_targets[~mask_pos_2]=0#[batch_size,h*w,1]
    #     # cnt_targets[~mask_pos_2]=-1
    #     reg_targets[~mask_pos_2]=-1
    #
    #     cnt_targets[mask_pos_2] = 1
    #     # cnt_targets[mask_cnt] = 1
    #     # print(cls_targets[cls_targets>0])
    #     theta_c_targets[~mask_pos_2]=0
    #     theta_r_targets[~mask_pos_2]=-1
    #     return cls_targets,cnt_targets,reg_targets,theta_c_targets,theta_r_targets
    #
    #     # return cls_targets, cnt_targets, reg_targets, theta_r_targets



def compute_cls_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    preds_reshape=[]
    class_num=preds[0].shape[1]
    mask=mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        # pred = torch.clamp(pred, 1e-4, 1.0 - 1e-4)
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,class_num])
        # print(pred.shape)
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)#[batch_size,sum(_h*_w),class_num]
    assert preds.shape[:2]==targets.shape[:2]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index]#[sum(_h*_w),class_num]
        target_pos=targets[batch_index]#[sum(_h*_w),1]
        # target_pos=(torch.arange(1,class_num+1,device=target_pos.device)[None,:]==target_pos).float()#sparse-->onehot
        # print(target_pos.shape)
        loss.append(focal_loss_from_logits(pred_pos,target_pos).view(1))
    # return torch.cat(loss, dim=0) / num_pos  # [batch_size,]
    #     loss.append(GaussLoss(pred_pos, target_pos).view(1))
    #     loss.append(qfloss(pred_pos,target_pos).view(1))
        # loss.append(smooth_focal_loss_from_logits(class_num,pred_pos,target_pos).view(1))

    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def compute_cnt_loss(preds,targets,mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    mask=mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        # pred = torch.clamp(pred, 1e-4, 1.0 - 1e-4)
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),1]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,]
        assert len(pred_pos.shape)==1
        loss.append(nn.functional.binary_cross_entropy_with_logits(input=pred_pos,target=target_pos,reduction='sum').view(1))
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]

def compute_reg_loss(preds,targets,mask,mode='smoothl1'):
    '''
    Args  
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]

    preds_reshape=[]
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos=torch.sum(mask,dim=1).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),4]
    loss=[]
    for batch_index in range(batch_size):

        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,4]
        # print(pred_pos)
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,4]
        # print(target_pos)
        assert len(pred_pos.shape)==2
        # if mode=='iou':
        #     loss.append(iou_loss(pred_pos,target_pos).view(1))
        # elif mode=='giou':
        #     loss.append(giou_loss(pred_pos,target_pos).view(1))
        # elif mode == 'smoothl1':
        # print(F.smooth_l1_loss(pred_pos,target_pos))
        loss.append(F.smooth_l1_loss(pred_pos,target_pos,reduction='sum').view(1))
        # else:
        #     raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]


def compute_att_loss(preds,targets,mask):
    '''
    Args
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    mask=mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    # num_pos=torch.sum(mask,dim=[1,2]).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        # pred = torch.clamp(pred, 1e-4, 1.0 - 1e-4)
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),1]
    loss=[]
    # print(pred)
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index]#[num_pos_b,]
        # print(pred_pos.shape)
        target_pos=targets[batch_index]#[num_pos_b,]
        with autocast(enabled=False):
            L = nn.functional.binary_cross_entropy(input=pred_pos, target=target_pos, reduction='mean')
        loss.append(L.view(1))
    return torch.cat(loss,dim=0)#[batch_size,]

def compute_theta_r_loss(preds,targets,mask,mode='smoothl1'):
    '''
    Args
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size=targets.shape[0]
    c=targets.shape[-1]
    preds_reshape=[]
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos=torch.sum(mask,dim=1).clamp_(min=1).float()#[batch_size,]
    for pred in preds:
        pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),4]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,4]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,4]
        assert len(pred_pos.shape)==2
        if mode=='iou':
            loss.append(iou_loss(pred_pos,target_pos).view(1))
        elif mode=='giou':
            loss.append(giou_loss(pred_pos,target_pos).view(1))
        elif mode == 'smoothl1':
            loss.append(F.smooth_l1_loss(pred_pos,target_pos,reduction='sum').view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]


def compute_theta_c_loss(preds,targets,mask):
    '''
       Args
       preds: list contains five level pred [batch_size,class_num,_h,_w]
       targets: [batch_size,sum(_h*_w),1]
       mask: [batch_size,sum(_h*_w)]
       '''
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    # mask = mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()  # [batch_size,]
    for pred in preds:
        # pred = torch.clamp(pred, 1e-4, 1.0 - 1e-4)
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, class_num])
        # print(pred.shape)
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)  # [batch_size,sum(_h*_w),class_num]
    assert preds.shape[:2] == targets.shape[:2]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
        target_pos = targets[batch_index]  # [sum(_h*_w),1]

        loss.append(focal_loss_from_logits(pred_pos, target_pos).view(1))


    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]

def iou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt=torch.min(preds[:,:2],targets[:,:2])
    rb=torch.min(preds[:,2:],targets[:,2:])
    wh=(rb+lt).clamp(min=0)
    overlap=wh[:,0]*wh[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    iou=overlap/(area1+area2-overlap)
    loss=-iou.clamp(min=1e-6).log()
    return loss.sum()

def giou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min=torch.min(preds[:,:2],targets[:,:2])
    rb_min=torch.min(preds[:,2:],targets[:,2:])
    wh_min=(rb_min+lt_min).clamp(min=0)
    overlap=wh_min[:,0]*wh_min[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    union=(area1+area2-overlap)
    iou=overlap/union

    lt_max=torch.max(preds[:,:2],targets[:,:2])
    rb_max=torch.max(preds[:,2:],targets[:,2:])
    wh_max=(rb_max+lt_max).clamp(0)
    G_area=wh_max[:,0]*wh_max[:,1]#[n]

    giou=iou-(G_area-union)/G_area.clamp(1e-10)
    loss=1.-giou
    return loss.sum()

def focal_loss_from_logits(preds,targets,gamma=2.0,alpha=0.25,smooth=True):
    '''
    Args:
    preds: [n,class_num] 
    targets: [n,class_num]
    '''
    label_smooth = 0.1
    class_num = preds.shape[-1]
    preds = preds.float()
    preds = preds.sigmoid()
    # if class_num == 1:class_num=2
    if class_num > 1 and smooth:
        preds = torch.clamp(preds.float(), min=label_smooth / (class_num - 1), max=1.0 - label_smooth)
    pt=preds*targets+(1.0-preds)*(1.0-targets)
    w=alpha*(1.0-targets)+(1.0-alpha)*targets
    FL = -w * torch.pow((1.0 - pt), gamma) * pt.log()
    epsilon = 1.0
    loss = FL + epsilon * torch.pow(1 - pt, gamma + 1)
    return loss.sum()

def smooth_focal_loss_from_logits(class_num,preds,targets,gamma=2.0,alpha=0.25):
    '''
    Args:
    preds: [n,class_num]
    targets: [n,class_num]
    '''
    label_smooth = 0.1

    preds = preds.sigmoid()
    preds = torch.clamp(preds.float(), min=label_smooth / (class_num - 1), max=1.0 - label_smooth)
    pt=preds*targets+(1.0-preds)*(1.0-targets)
    w=alpha*(1.0-targets)+(1.0-alpha)*targets
    FL =-w*torch.pow((1.0-pt),gamma)*pt.log()
    epsilon = 1.0
    loss = FL + epsilon * torch.pow(1 - pt, gamma + 1)
    return loss.sum()
# from dataloader.dota_dataset import classes_weights
# nw = classes_weights()

def qfloss(preds, target, gamma=1.5,alpha=0.25):

    # binary_cross_entropy_with_logits = sigmoid(preds) + binary_cross_entropy()

    # loss = F.binary_cross_entropy_with_logits(preds, target, reduction='none',weight=nw.to(device = preds.device))
    loss = F.binary_cross_entropy_with_logits(preds, target, reduction='none')
    pred_prob = torch.sigmoid(preds)  # prob from logits
    alpha_factor = target * alpha + (1 - target) * (1 - alpha)
    modulating_factor = torch.abs(target - pred_prob) ** gamma
    loss *= alpha_factor * modulating_factor


    # nums = (target>0).sum()
    # if nums == 0:
    #     loss = loss.sum()
    # else:
    #     loss = loss.sum()/ (1e-4 + nums)
    return loss.sum()

# Pytorch


class LOSS(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
        self.Refine = RefinedLoss()
        self.BBD = BoxCoder()
    def forward(self,inputs):

        preds,targets=inputs
        full_cls_logits,full_cnt_logits,full_reg_preds,full_tc_preds,full_tr_preds=preds
        # for i in cnt_logits:
        #     print(i.shape)
        cls_logits, cnt_logits, reg_preds,tc_preds,tr_preds = full_cls_logits[0:], full_cnt_logits[0:], full_reg_preds[0:],full_tc_preds[0:],full_tr_preds
        cls_targets, cnt_targets, reg_targets,tc_targets,tr_targets = targets['P5_7']
        # cls_logits, cnt_logits, reg_preds = full_cls_logits[1:],full_cnt_logits[1:],full_reg_preds[1:]
        # cls_targets,cnt_targets,reg_targets = targets['P5_7']

        # P2cls_logits, P2cnt_logits, P2reg_preds = full_cls_logits[0], full_cnt_logits[0], full_reg_preds[0]
        # P2cls_targets, P2cnt_targets, P2reg_targets = targets['P2']



        mask_pos = (cnt_targets > 0).squeeze(dim=-1)  # [batch_size,sum(_h*_w)]
        cls_loss = compute_cls_loss(cls_logits,cls_targets,mask_pos).mean()#[]
        # cls_loss = GaussLoss(pred=cls_logits, target=cls_targets).mean()
        cnt_loss = compute_att_loss(cnt_logits,cnt_targets,mask_pos).mean()
        reg_loss=compute_reg_loss(reg_preds,reg_targets,mask_pos).mean()

        tc_loss = compute_theta_c_loss(tc_preds,tc_targets,mask_pos).mean()
        tr_loss = compute_theta_r_loss(tr_preds,tr_targets,mask_pos).mean()

        # mask_p2 = P2cnt_targets.permute(0,2,3,1) > 0.945 # b h w
        #
        # mask_p2 = mask_p2.squeeze(dim=-1)
        # P2_cls_loss = GaussLoss(pred=P2cls_logits,target=P2cls_targets).mean()
        # P2_reg_loss = gauss_reg_loss(pred=P2reg_preds,target=P2reg_targets,mask=mask_p2).mean()
        # P2_cnt_loss = Gauss_attLoss(pred=P2cnt_logits,target=P2cnt_targets).mean()

        P2_cls_loss = tc_loss
        P2_reg_loss = tr_loss
        P2_cnt_loss = 0



        total_loss = cls_loss + reg_loss + cnt_loss + P2_cls_loss + P2_reg_loss + P2_cnt_loss
        return cls_loss, cnt_loss, reg_loss,P2_cls_loss , P2_reg_loss , P2_cnt_loss,total_loss
        # total_loss = cls_loss + reg_loss + cnt_loss
        # return cls_loss, cnt_loss, reg_loss,0,0,0, total_loss






if __name__=="__main__":
    loss=compute_cnt_loss([torch.ones([2,1,4,4])]*5,torch.ones([2,80,1]),torch.ones([2,80],dtype=torch.bool))
    print(loss)




        


        































