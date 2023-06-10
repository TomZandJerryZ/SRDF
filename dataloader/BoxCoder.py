# from model.loss import coords_fmap2orig
import torch
from model.config import DefaultConfig
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from overlaps.rbox_overlaps import rbox_overlaps
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
def min_area_square(rboxes):
    w = rboxes[..., 2] - rboxes[..., 0]
    h = rboxes[..., 3] - rboxes[..., 1]
    ctr_x = rboxes[..., 0] + w * 0.5
    ctr_y = rboxes[..., 1] + h * 0.5
    s = torch.max(w, h)
    return torch.stack((
        ctr_x - s * 0.5, ctr_y - s * 0.5,
        ctr_x + s * 0.5, ctr_y + s * 0.5),
        dim=1
    )
def bbox_overlaps(boxes, query_boxes):
    if not isinstance(boxes,float):   # apex
        boxes = boxes.float()
    area = (query_boxes[..., 2] - query_boxes[..., 0]) * \
           (query_boxes[..., 3] - query_boxes[..., 1])
    iw = torch.min(torch.unsqueeze(boxes[..., 2], dim=1), query_boxes[..., 2]) - \
         torch.max(torch.unsqueeze(boxes[..., 0], 1), query_boxes[..., 0])
    ih = torch.min(torch.unsqueeze(boxes[..., 3], dim=1), query_boxes[..., 3]) - \
         torch.max(torch.unsqueeze(boxes[..., 1], 1), query_boxes[..., 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    return intersection / ua


def rbox_overlaps_(boxes, query_boxes, indicator=None, thresh=1e-1):
    # rewrited by cython
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    a_tt = boxes[:, 4]
    a_ws = boxes[:, 2] - boxes[:, 0]
    a_hs = boxes[:, 3] - boxes[:, 1]
    a_xx = boxes[:, 0] + a_ws * 0.5
    a_yy = boxes[:, 1] + a_hs * 0.5

    b_tt = query_boxes[:, 4]
    b_ws = query_boxes[:, 2] - query_boxes[:, 0]
    b_hs = query_boxes[:, 3] - query_boxes[:, 1]
    b_xx = query_boxes[:, 0] + b_ws * 0.5
    b_yy = query_boxes[:, 1] + b_hs * 0.5

    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = b_ws[k] * b_hs[k]
        for n in range(N):
            if indicator is not None and indicator[n, k] < thresh:
                continue
            ua = a_ws[n] * a_hs[n] + box_area
            rtn, contours = cv2.rotatedRectangleIntersection(
                ((a_xx[n], a_yy[n]), (a_ws[n], a_hs[n]), a_tt[n]),
                ((b_xx[k], b_yy[k]), (b_ws[k], b_hs[k]), b_tt[k])
            )
            if rtn == 1:
                ia = cv2.contourArea(contours)
                overlaps[n, k] = ia / (ua - ia)
            elif rtn == 2:
                ia = np.minimum(ua - box_area, box_area)
                overlaps[n, k] = ia / (ua - ia)
    return overlaps

class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def __init__(self,full=True):
        if full:
            self.strides = DefaultConfig.strides
        else:
            self.strides = DefaultConfig.strides[1:]
    def _reshape_cat_out_reg(self,inputs,strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            pred = pred.permute(0,2,3,1)
            # coord = coords_fmap2orig(pred,stride).to(device=pred.device)
            pred = torch.reshape(pred,[batch_size,-1,c])
            pred[:,:,0:4] = pred[:,:,0:4] * stride
            out.append(pred)

        return torch.cat(out,dim=1)
    def _reshape_cat_out(self,inputs,strides):

        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            # print(pred.shape)
            pred = pred.permute(0,2,3,1)
            coord = coords_fmap2orig(pred,stride).to(device=pred.device)
            pred = torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0) # B N 5,B N 2
    def IBO_cv2(self,theta):
        mask_IBO = theta > 90
        theta[mask_IBO] = theta[mask_IBO] - 180
        theta = theta * -1
        return theta
    def refine_bbx_clstheta(self, out,inputs):

        cls, _, reg_preds = out
        # cls_targets, att_t, reg_targets = inputs
        _,_,reg_p57 = inputs['P5_7']
        _,_,reg_p2 = inputs['P2']
        reg_p2 = reg_p2.reshape(reg_p2.shape[0],-1,5)



        reg_targets = torch.cat([reg_p2,reg_p57],dim=1)
        # print(reg_targets.shape)
        cls_logits, coords = self._reshape_cat_out(cls, self.strides)  # [batch_size,sum(_h*_w),class_num]
        reg_preds,_ = self._reshape_cat_out(reg_preds, self.strides)
        # print(reg_preds.shape,reg_targets.shape)



        x1y1 = coords[None, :, :] - reg_preds[..., :2]
        x2y2 = coords[None, :, :] + reg_preds[..., 2:-1]
        angle = reg_preds[..., -1]
        pred_theta = angle / 15 * 180.0
        pred_theta = self.IBO_cv2(pred_theta)
        boxes = torch.cat([x1y1, x2y2, pred_theta.unsqueeze(-1)], dim=-1)

        T_theta = reg_targets[...,-1]
        T_theta = T_theta / 15 * 180.0
        T_theta = self.IBO_cv2(T_theta)
        T_x1y1 = coords[None, :, :] - reg_targets[..., :2]
        T_x2y2 = coords[None, :, :] + reg_targets[..., 2:-1]

        T_boxes = torch.cat([T_x1y1, T_x2y2, T_theta.unsqueeze(-1)], dim=-1)



        return [boxes ,cls_logits,reg_preds,T_boxes] # [batch_size,sum(_h*_w),5] ,# [batch_size,sum(_h*_w),numcls]

from locat_overlaps.locat_overlaps import locat_overlaps

class RefinedLoss(nn.Module):
    def __init__(self):
        super(RefinedLoss, self).__init__()


    def forward(self, pred, target, iou_thres=0.4):
        P_bbx, cls_logits,reg_preds, T_boxes = pred

        # cls_targets, att_t, reg_targets = target
        # cls_targets, cnt_targets, reg_targets = target

        _, cnt_p57, reg_p57 = target['P5_7']
        _, cnt_p2, reg_p2 = target['P2']
        # print(cnt_p57.shape)
        reg_p2 = reg_p2.reshape(reg_p2.shape[0], -1, 5)

        reg_targets = torch.cat([reg_p2, reg_p57], dim=1)
        cnt_p2 = cnt_p2.reshape(cnt_p2.shape[0],-1,1)
        cnt_targets = torch.cat([cnt_p2,cnt_p57],dim=1)
        refine_reg_losses = []
        batch_size = cls_logits.shape[0]
        for j in range(batch_size):
            P_bbx_ = P_bbx[j]
            mask_t = cnt_targets[j]
            target_box = T_boxes[j]
            iou = locat_overlaps(
                P_bbx_.detach().cpu().numpy(),
                target_box.cpu().numpy(),
                mask_t.cpu().numpy()
            )
            if not torch.is_tensor(iou):
                iou = torch.from_numpy(iou).cuda()
            positive_indices = iou > iou_thres

            if positive_indices.sum() > 0:

                num_pos = positive_indices.sum()
                reg_pred_pos = reg_preds[j][positive_indices]
                reg_target_pos = reg_targets[j][positive_indices]
                refine_loss = F.smooth_l1_loss(reg_pred_pos, reg_target_pos, reduction='sum').view(1)/num_pos
                refine_reg_losses.append(refine_loss)

            else:
                refine_reg_losses.append(torch.tensor([0]).float().cuda())


        loss_refine_reg = torch.stack(refine_reg_losses).mean(dim=0, keepdim=True)


        return loss_refine_reg

