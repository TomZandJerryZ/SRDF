import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from .attention import ChannelAttention,SpatialAttention,Attention
# from .Swin_T import SwinTransformerBlock
from .config import DefaultConfig


class ClsCntRegHead(nn.Module):
    def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
        '''
        Args  
        in_channel  
        class_num  
        GN  
        prior  
        '''
        super(ClsCntRegHead,self).__init__()
        self.prior=prior
        self.class_num=class_num
        self.cnt_on_reg=cnt_on_reg
        self.Att = Attention()
        self.CAtt = ChannelAttention()
        self.SA = SpatialAttention()



        cls_branch=[]
        reg_branch=[]
        cnt_branch=[]
        theta_branch = []

        for i in range(4):
            cls_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=1, bias=True))
            if GN:
                cls_branch.append(nn.GroupNorm(32, in_channel))
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=1, bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32, in_channel))
            reg_branch.append(nn.ReLU(True))


            cnt_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=1, bias=True))
            if GN:
                cnt_branch.append(nn.GroupNorm(32, in_channel))
            cnt_branch.append(nn.ReLU(True))

            theta_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=1, bias=True))
            if GN:
                theta_branch.append(nn.GroupNorm(32, in_channel))
            theta_branch.append(nn.ReLU(True))



        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)
        self.cnt_conv = nn.Sequential(*cnt_branch)

        self.theta_conv = nn.Sequential(*theta_branch)

        self.sigmoid = nn.Sigmoid()
        self.sigmoid_t = nn.Sigmoid()

        self.cls_logits=nn.Conv2d(in_channel,class_num,kernel_size=(3,3),padding=1)

        self.reg_pred=nn.Conv2d(in_channel,4,kernel_size=(3,3),padding=1)

        self.conv_t= nn.Conv2d(in_channel, 1,kernel_size=(3,3),padding=1)

        self.theta_cls_lo = nn.Conv2d(in_channel,18,kernel_size=(3,3),padding=1)
        self.theta_reg_lo = nn.Conv2d(in_channel,1,kernel_size=(3,3),padding=1)
        self.apply(self.init_conv_RandomNormal)
        
        nn.init.constant_(self.cls_logits.bias,-math.log((1 - prior) / prior))

        nn.init.constant_(self.theta_cls_lo.bias, -4.19)

    
    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,inputs):
        '''inputs:[P3~P7]'''
        cls_logits=[]
        cnt_logits=[]
        reg_preds=[]

        theta_cls_lo = []
        theta_reg_lo = []

        for index,P in enumerate(inputs):

            cnt_conv_out = self.Att(self.cnt_conv(P))
            cnt_conv_out = self.sigmoid(cnt_conv_out)
            P = self.CAtt(P) * P
            P = P * self.SA(P)
            p = cnt_conv_out * P

            cls_conv_out = self.cls_conv(p)
            reg_conv_out = self.reg_conv(p)

            theta_conv_out = self.theta_conv(p)


            cls_logits.append(self.cls_logits(cls_conv_out))
            reg_size = self.reg_pred(reg_conv_out)
            theta = self.conv_t(reg_conv_out).float()
            angle = self.sigmoid_t(theta) * DefaultConfig.T_a
            reg_pr = torch.cat((reg_size, angle), 1)
            reg_preds.append(reg_pr)
            cnt_logits.append(cnt_conv_out)

            theta_cls = self.theta_cls_lo(theta_conv_out)
            theta_reg = self.theta_reg_lo(theta_conv_out)
            theta_cls_lo.append(theta_cls)
            theta_reg_lo.append(theta_reg)



        return cls_logits,cnt_logits,reg_preds, theta_cls_lo,theta_reg_lo




        
