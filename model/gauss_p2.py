import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from .config import DefaultConfig
# 制作标签

def np_coords_fmap4(h,w,stride=1):
    # h,w=feature.shape[1:3]
    shifts_x = np.arange(0, w * stride, stride, dtype=np.float32)
    shifts_y = np.arange(0, h * stride, stride, dtype=np.float32)

    shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])
    coords = np.stack([shift_y, shift_x], -1)
    return coords

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
# 模拟一下
# 中心点为 25,40的点 长10 宽6

label = [20,43,30,37]

class Gauss:
    def __init__(self,class_num,fea_h,fea_w):
        self.class_num = class_num
        self.fea_h = fea_h
        self.fea_w = fea_w
        self.coords = np_coords_fmap4(self.fea_h,self.fea_w).reshape(self.fea_h,self.fea_w,2)


    def gauss_Donut(self,cls_map,reg_map,gt_label,cls):
        # 这里需要一个特征图尺寸
        #
        # cls_map_temp = np.zeros((self.class_num, self.fea_h, self.fea_w), dtype=np.float32)
        n = gt_label.shape[0]
        for i in range(n):
            cls_map_temp = np.zeros((self.fea_h, self.fea_w), dtype=np.float32)
            x1, y1, x2, y2 , t = gt_label[i]
            x1, y1, x2, y2 = x1/4, y1/4, x2/4, y2 /4
            cls_label = cls[i] - 1
            ct_x ,ct_y = (x1+x2)//2 , (y1+y2)//2
            # print(ct_x ,ct_y)
            # gs_r = min((x2 - x1), (y2 - y1)) / 4  # 高斯核半径
            # gs_x = self.coords[:, :, 0] - ct_x
            # gs_y = self.coords[:, :, 1] - ct_y
            biu = np.zeros_like(self.coords[:, :, 0])
            targets_dt = (t / 180) * DefaultConfig.T_a
            targets_dt = targets_dt - biu
            l_off = self.coords[:, :, 0] - x1
            t_off = self.coords[:, :, 1] - y1
            r_off = x2 - self.coords[:, :, 0]
            b_off = y2 - self.coords[:, :, 1]
            biu_off = np.stack([l_off, t_off, r_off, b_off, targets_dt], axis=-1)  # [h,w,5]
            # if D:
            #     gauss_D = np.exp(-((gs_x) ** 2 + (gs_y) ** 2) / (2 * gs_r * gs_r + 1e-5))
            #     gauss_D[gauss_D < 0.09] = 0
            #     cls_map_temp = gauss_D
            # else:
            radius = gaussian_radius((math.ceil(x2 - x1), math.ceil(y2 - y1)))
            radius = max(0, int(radius))
            cls_map_temp = draw_umich_gaussian(cls_map_temp, np.array([ct_x, ct_y]), radius)




            # plt.imshow(cls_map_temp)
            # plt.show()


            mask = cls_map_temp > 0

            cls_map[int(cls_label)][mask] = cls_map_temp[mask]
            reg_map[mask] = biu_off[mask]

        # print('rm',reg_map.shape)

        # m1 = cls_map == 1.0
        # if n != m1.sum():
        cnt_map = np.sum(cls_map, axis=0)
        # plt.imshow(cnt_map)
        # plt.show()
        cnt_map[cnt_map>0.1] = 1
        cnt_map[cnt_map<=0.1] = 0
        cnt_map = cnt_map[np.newaxis,:]
        # print(c.shape)


        # assert m1.sum() == n

        return cls_map,reg_map ,cnt_map


def batch_gen_gausstarget(batch_gt,batch_cls,level_out,stride,limit_range):
    # print(stride)
    cls_logits, _, _ = level_out
    device = cls_logits.device
    batch_size = cls_logits.shape[0]
    class_num = cls_logits.shape[1]
    fea_h ,fea_w = cls_logits.shape[2:]
    cls_logits = cls_logits.permute(0, 2, 3, 1)  # [batch_size,h,w,class_num]

    cls_logits = cls_logits.reshape((batch_size, -1, class_num))  # [batch_size,h*w,class_num]
    h_mul_w = cls_logits.shape[1]
    gt = batch_gt.cpu().numpy() # b m 5
    cls = batch_cls.cpu().numpy() # b m

    batch_cls_targets = []
    batch_cnt_targets = []
    batch_reg_targets = []

    if stride == 4:
        G = Gauss(class_num=class_num,fea_h=fea_h,fea_w=fea_w)
        for j in range(batch_size):
            cls_map = np.zeros((class_num, fea_h, fea_w), dtype=np.float32)
            reg_map = np.zeros((fea_h, fea_w,5), dtype=np.float32)
            gt_box = gt[j, :, :]
            cls_b = cls[j, :]
            area = (gt_box[:, 3] - gt_box[:, 1]) * (gt_box[:, 2] - gt_box[:, 0])
            mask_level = (area > limit_range[0]) & (area <= limit_range[1])
            if mask_level.sum() != 0:

                cls_targets, reg_targets ,cnt_targets = G.gauss_Donut(cls_map,reg_map,gt_box[mask_level],cls_b[mask_level])

            else:
                cls_targets = np.zeros((class_num, fea_h,fea_w)).astype(np.float32)
                reg_targets = -np.ones((fea_h,fea_w, 5)).astype(np.float32)
                cnt_targets = np.zeros((1, fea_h,fea_w)).astype(np.float32)
            batch_cls_targets.append(cls_targets)
            batch_cnt_targets.append(cnt_targets)
            batch_reg_targets.append(reg_targets)

    batch_cls_targets = np.stack(batch_cls_targets, axis=0)
    batch_cnt_targets = np.stack(batch_cnt_targets, axis=0)
    batch_reg_targets = np.stack(batch_reg_targets, axis=0)

    return torch.from_numpy(batch_cls_targets).to(device), torch.from_numpy(batch_cnt_targets).to(
        device), torch.from_numpy(batch_reg_targets).to(device)



