import torch.nn as nn
import torch.nn.functional as F
import math
from .deeplabv3 import ASPP_Bottleneck

from .Swin_T import SwinTransformerBlock


class FPN(nn.Module):
    '''only for resnet50,101,152'''
    def __init__(self,features=256,use_p5=True):
        super(FPN,self).__init__()
        self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.conv_5 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.prj_2 = nn.Conv2d(256, features, kernel_size=1)
        self.conv_2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.Dconv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.Dconv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.Dconv_out5 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.Dconv_out4 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.Dconv_out3 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5=use_p5
        self.apply(self.init_conv_kaiming)
        num_heads = features // 32
        n = 1
        self.m = SwinTransformerBlock(features, features, num_heads, n)
        # self.aspp = ASPP_Bottleneck()
    def upsamplelike(self,inputs):
        src,target=inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                    mode='nearest')
    
    def init_conv_kaiming(self,module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,x):
        C2,C3,C4,C5=x
        # print(C3.shape,C4.shape,C5.shape)
        P5 = self.prj_5(C5)
        # P5 = self.aspp(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        P2 = self.prj_2(C2)
        
        P4 = P4 + self.upsamplelike([P5,C4])
        P3 = P3 + self.upsamplelike([P4,C3])

        P2 = P2 + self.upsamplelike([P3,C2])
        P3 = self.conv_3(P3)
        P2 = self.conv_2(P2)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        
        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        # return [P3, P4, P5, P6, P7]
        # D2 = self.m(P2)
        D2 = self.conv_2(P2)
        # D2 = P2
        D3 = self.Dconv_out3(D2) + P3
        D4= self.Dconv_out4(D3) + P4
        D5 = self.Dconv_out5(D4) + P5
        D6 = self.Dconv_out6(D5) + P6
        D7 = self.Dconv_out7(D6) + P7

        # return [P2,P5,P6,P7]
        return [D2,D3,D4, D5, D6,D7]


if __name__ == '__main__':
    from backbone.resnet import resnet50
    import torch

    backbone = resnet50(pretrained=True, if_include_top=False)
    fpn = FPN(256, use_p5=True)
    c = torch.randn(3, 3, 608, 608)

    # print(DL(c).shape)
    for i in fpn(backbone(c)):
        print(i.shape)
