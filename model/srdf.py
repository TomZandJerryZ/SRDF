from .head import ClsCntRegHead
from .fpn import FPN
from .backbone.resnet import resnet50,resnet101
import torch.nn as nn
from .loss import GenTargets,LOSS,coords_fmap2orig
import torch
from .config import DefaultConfig
from .deeplabv3 import DeepLabV3
import numpy as np
from dataloader.BoxCoder import *

class SRDF(nn.Module):
    
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        # self.backbone=resnet50(pretrained=config.pretrained,if_include_top=False)
        self.backbone = resnet101(pretrained=config.pretrained, if_include_top=False)
        self.fpn=FPN(config.fpn_out_channels,use_p5=config.use_p5)
        # self.fpn = DeepLabV3()
        self.head=ClsCntRegHead(config.fpn_out_channels,config.class_num,
                                config.use_GN_head,config.cnt_on_reg,config.prior)
        self.config=config
    def train(self,mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False
        if self.config.freeze_bn:
            self.apply(freeze_bn)
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)

    def forward(self,x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C2 ,C3,C4,C5=self.backbone(x)
        all_P=self.fpn([C2,C3,C4,C5])

        cls_logits,cnt_logits,reg_preds,tc_preds,tr_preds=self.head(all_P)
        return [cls_logits,cnt_logits,reg_preds,tc_preds,tr_preds]

class DetectHead(nn.Module):
    def __init__(self,score_threshold,nms_iou_threshold,max_detection_boxes_num,strides,config=None):
        super().__init__()
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs):



        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]

        reg_preds = self._reshape_cat_out_reg(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds=cls_logits.sigmoid_()

        cls_scores,cls_classes=torch.max(cls_preds,dim=-1)#[batch_size,sum(_h*_w)]

        cls_classes=cls_classes+1#[batch_size,sum(_h*_w)]

        tc_lo,_ = self._reshape_cat_out(inputs[3], self.strides)
        tr_lo,_ = self._reshape_cat_out(inputs[4], self.strides)
        # print(tr_lo.shape)

        tc_preds = tc_lo.sigmoid_()
        # print(tc_preds.shape)

        t_scores, t_classes = torch.max(tc_preds, dim=-1)  # [batch_size,sum(_h*_w)]

        theta_c = t_classes+1 # [batch_size,sum(_h*_w)]
        theta_c = theta_c.unsqueeze(-1)
        theta = theta_c*10+tr_lo

        whc = self._coords2boxes(coords,reg_preds,theta)

        max_num=min(self.max_detection_boxes_num,cls_scores.shape[-1])
        topk_ind=torch.topk(cls_scores,max_num,dim=-1,largest=True,sorted=True)[1]#[batch_size,max_num]
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]

        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])#[max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])#[max_num]
            _boxes.append(whc[batch][topk_ind[batch]])#[max_num,4]

        cls_scores_topk=torch.stack(_cls_scores,dim=0)#[batch_size,max_num]

        cls_classes_topk=torch.stack(_cls_classes,dim=0)#[batch_size,max_num]

        boxes_topk=torch.stack(_boxes,dim=0)#[batch_size,max_num,4]

        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self,preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''

        from nms_wrapper import nms
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        _theta_post=[]
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk

        for batch in range(cls_classes_topk.shape[0]):
            mask=cls_scores_topk[batch]>=self.score_threshold
            _cls_scores_b=cls_scores_topk[batch][mask]#[?]
            _cls_classes_b=cls_classes_topk[batch][mask]#[?]
            _boxes_b=boxes_topk[batch][mask]#[?,4]

            anchors_nms_idx = nms(torch.cat([_boxes_b,_cls_scores_b.unsqueeze(-1),_cls_classes_b.unsqueeze(-1)], dim=-1), self.nms_iou_threshold)

            _cls_scores_post.append(_cls_scores_b[anchors_nms_idx])
            _cls_classes_post.append(_cls_classes_b[anchors_nms_idx])
            _boxes_post.append(_boxes_b[anchors_nms_idx,:])
            # _theta_post.append(_theta_b)
        scores,classes,boxes= torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)
        # thetas = torch.stack(_theta_post,dim=0)
        return scores.data.cpu().numpy(),classes.data.cpu().numpy(),boxes.data.cpu().numpy()




    def _coords2boxes(self, coords, offsets, theta):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:-1]  # [batch_size,sum(_h*_w),2]

        angle = offsets[..., -1]
        pred_theta = angle / DefaultConfig.T_a * 180.0

        # print(pred_theta)
        # print(theta)
        # boxes = torch.cat([x1y1, x2y2, pred_theta.unsqueeze(-1)], dim=-1)
        # print(theta.shape)

        boxes = torch.cat([x1y1, x2y2, theta], dim=-1)


        return boxes


    def _reshape_cat_out(self,inputs,strides):
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
            # print(stride)
            pred = pred.permute(0,2,3,1)
            coord = coords_fmap2orig(pred,stride).to(device=pred.device)
            pred = torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0)
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

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes

        
class SRDFDetector(nn.Module):
    def __init__(self,mode="training",config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.mode=mode
        self.srdf_body=SRDF(config=config)
        self.Refine = RefinedLoss()
        self.BBD = BoxCoder()
        if mode=="training":
            self.target_layer=GenTargets(strides=config.strides,limit_range=config.limit_range,gauss_range=config.gauss_range)
            self.loss_layer=LOSS()
        elif mode=="inference":
            self.detection_head=DetectHead(config.score_threshold,config.nms_iou_threshold,
                                            config.max_detection_boxes_num,config.strides,config)
            self.clip_boxes=ClipBoxes()
        
    
    def forward(self,inputs):
        '''
        inputs 
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''

        if self.mode=="training":
            batch_imgs,batch_boxes,batch_cls=inputs
            out=self.srdf_body(batch_imgs)
            targets=self.target_layer([out,batch_boxes,batch_cls])
            cls_loss, cnt_loss, reg_loss,P2_cls_loss , P2_reg_loss , P2_cnt_loss,total_loss = self.loss_layer([out,targets])
            # preds = self.BBD.refine_bbx_clstheta(out, targets)
            # loss_refine_reg = self.Refine(preds, targets)
            # refine_totalloss = total_loss + loss_refine_reg
            refine_totalloss = total_loss
            return cls_loss, cnt_loss, reg_loss,0,P2_cls_loss , P2_reg_loss , P2_cnt_loss,refine_totalloss
        elif self.mode=="inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net 
            '''
            batch_imgs=inputs
            out=self.srdf_body(batch_imgs)
            scores,classes,boxes=self.detection_head(out)
            # boxes=self.clip_boxes(batch_imgs,boxes)
            return scores[0],classes[0],boxes[0]



    


