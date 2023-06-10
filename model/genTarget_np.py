import numpy as np
import torch

from dataloader.bbox import quad_2_rbox,rbox_2_quad,sort_corners
def in_gbox(gt_boxess, coords):
    gt_boxes = gt_boxess
    gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2] + gt_boxes[..., 4] + gt_boxes[..., 6]) / 4
    gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3] + gt_boxes[..., 5] + gt_boxes[..., 7]) / 4
    gt_center = np.concatenate([gt_center_x[:,np.newaxis], gt_center_y[:,np.newaxis]], axis=-1)

    # print(gt_boxes.shape)
    # print(gt_center.shape)
    alpha = 0.2
    gt_boxes[..., 0:2] = gt_boxes[..., 0:2] - (gt_boxes[..., 0:2] - gt_center) * alpha
    gt_boxes[..., 2:4] = gt_boxes[..., 2:4] - (gt_boxes[..., 2:4] - gt_center) * alpha
    gt_boxes[..., 4:6] = gt_boxes[..., 4:6] - (gt_boxes[..., 4:6] - gt_center) * alpha
    gt_boxes[..., 6:] = gt_boxes[..., 6:] - (gt_boxes[..., 6:] - gt_center) * alpha

    # biu = torch.zeros_like(coords).to(device=gt_boxess.device)
    biu = np.zeros_like(coords)


    # print(gt_boxes[0,0,:],gt_boxess[0,0,:])
    # print((gt_boxes[..., 2:4] - gt_boxes[..., 0:2]).shape)
    AB = (gt_boxes[..., 2:4] - gt_boxes[..., 0:2])[None, :] - biu[:, None]
    BC = (gt_boxes[..., 4:6] - gt_boxes[..., 2:4])[None, :] - biu[:, None]
    CD = (gt_boxes[..., 6:] - gt_boxes[..., 4:6])[None, :] - biu[:, None]
    DA = (gt_boxes[..., 0:2] - gt_boxes[..., 6:])[None, :] - biu[:, None]

    # print(AB)
    AM = coords[:, None] - gt_boxes[..., 0:2][None, :]
    BM = coords[:, None] - gt_boxes[..., 2:4][None, :]
    CM = coords[:, None] - gt_boxes[..., 4:6][None, :]
    DM = coords[:, None] - gt_boxes[..., 6:][None, :]

    # [b,h*w,m,2]
    AB_AM = AB[:, :, 0] * AM[:, :, 0] + AB[:, :, 1] * AM[:, :, 1]
    BC_BM = BC[:, :, 0] * BM[:, :, 0] + BC[:, :, 1] * BM[:, :, 1]
    CD_CM = CD[:, :, 0] * CM[:, :, 0] + CD[:, :, 1] * CM[:, :, 1]
    DA_DM = DA[:, :, 0] * DM[:, :, 0] + DA[:, :, 1] * DM[:, :, 1]
    # print(AB_AM.shape)

    mask_in_gt = (AB_AM > 0) & (BC_BM > 0) & (CD_CM > 0) & (DA_DM > 0)
    return mask_in_gt


def np_coords_fmap2orig(feature,stride):
    '''
    numpy one f map coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    '''
    h,w=feature.shape[1:3]
    shifts_x = np.arange(0, w * stride, stride, dtype=np.float32)
    shifts_y = np.arange(0, h * stride, stride, dtype=np.float32)

    shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])
    coords = np.stack([shift_y, shift_x], -1) + stride // 2
    return coords


def five_to_eight(gt_boxes):
    eight = []
    for i in range(0, gt_boxes.shape[0]):
        dets = gt_boxes[i]
        res = sort_corners(rbox_2_quad(dets))
        eight.append(res[0])

    return np.stack(eight, axis=0)

from .gauss_p2 import Gauss


def batch_gen_target(batch_gt,batch_cls,level_out,stride,limit_range,gauss_range):
    # print(stride)
    cls_logits, _, _ = level_out
    device = cls_logits.device
    batch_size = cls_logits.shape[0]
    class_num = cls_logits.shape[1]
    fea_h ,fea_w = cls_logits.shape[2:]
    cls_logits = cls_logits.permute(0, 2, 3, 1)  # [batch_size,h,w,class_num]
    coords = np_coords_fmap2orig(cls_logits, stride)  # [h*w,2]

    cls_logits = cls_logits.reshape((batch_size, -1, class_num))  # [batch_size,h*w,class_num]
    h_mul_w = cls_logits.shape[1]
    gt = batch_gt.cpu().numpy() # b m 5
    cls = batch_cls.cpu().numpy() # b m

    batch_cls_targets = []
    batch_cnt_targets = []
    batch_reg_targets = []
    batch_tcls_targets = []
    batch_treg_targets = []

    for j in range(batch_size):
        gt_box = gt[j,:,:]
        cls_b = cls[j,:]

        area = (gt_box[:, 3] - gt_box[:, 1]) * (gt_box[:, 2] - gt_box[:, 0])
        mask_level = (area > limit_range[0]) & (area <= limit_range[1])
        if mask_level.sum() != 0:

            cls_targets, cnt_targets, reg_targets,tc_targets,tr_targets = np_gen_level_targets(coords,gt_box[mask_level],cls_b[mask_level],stride,gauss_range)
        else:
            cls_targets = np.zeros((h_mul_w,DefaultConfig.class_num)).astype(np.float32)

            cnt_targets = np.zeros((h_mul_w, 1)).astype(np.float32)
            reg_targets = -np.ones((h_mul_w, 5)).astype(np.float32)

            tc_targets = np.zeros((h_mul_w, 18)).astype(np.float32)
            tr_targets = -np.ones((h_mul_w, 1)).astype(np.float32)


        batch_cls_targets.append(cls_targets)
        batch_cnt_targets.append(cnt_targets)
        batch_reg_targets.append(reg_targets)

        batch_tcls_targets.append(tc_targets)
        batch_treg_targets.append(tr_targets)

    batch_cls_targets = np.stack(batch_cls_targets, axis=0)
    batch_cnt_targets = np.stack(batch_cnt_targets, axis=0)
    batch_reg_targets = np.stack(batch_reg_targets, axis=0)

    batch_tcls_targets = np.stack(batch_tcls_targets,axis=0)
    batch_treg_targets = np.stack(batch_treg_targets,axis=0)


    return torch.from_numpy(batch_cls_targets).to(device),torch.from_numpy(batch_cnt_targets).to(device),torch.from_numpy(batch_reg_targets).to(device),\
torch.from_numpy(batch_tcls_targets).to(device),torch.from_numpy(batch_treg_targets).to(device)

from .config import DefaultConfig

def np_gen_level_targets(coords,gt_boxes,classes,stride,gauss_range):#,sample_radiu_ratio=2


    x, y = coords[:, 0], coords[:, 1]
    # print(gt_boxes)
    eight_box = five_to_eight(gt_boxes)
    mig = in_gbox(eight_box, coords)
    biu = np.zeros_like(x)
    gt_center_x = (gt_boxes[..., 2] + gt_boxes[..., 0]) / 2
    gt_center_y = (gt_boxes[..., 3] + gt_boxes[..., 1]) / 2

    theta = gt_boxes[..., -1]  # [batch_size,m]
    # ex_thetas = np.zeros_like(theta)

    # targets_dt = 15 * (np.tan((45 - theta) / 180.0 * np.pi) - np.tan(ex_thetas / 180.0 * np.pi))
    # 180
    targets_dt = (theta /180) * DefaultConfig.T_a
    targets_dt = targets_dt[None, :] - biu[:, None]

    theta_cls = theta // 10 # 18
    theta_reg = theta - theta_cls * 10
    theta_cls = theta_cls[None, :] - biu[:, None]

    theta_reg = theta_reg[None, :] - biu[:, None]

    theta_reg = theta_reg[:,:,np.newaxis]

    # print(theta_reg.shape)
    # print(theta)
    # print(theta_cls)
    # print(theta_reg)


    classes = classes[None, :] - biu[:, None]
    # print(classes)
    w = gt_boxes[..., 2] - gt_boxes[..., 0]
    h = gt_boxes[..., 3] - gt_boxes[..., 1]
    w_h = np.stack([w,h],axis=-1) / 2
    # print(w_h)
    # print(gt_center_y.shape)
    short_size = np.min(w_h,axis=-1)
    # print(short_size.shape)
    gs_x = x[:, None] - gt_center_x[None,:]
    gs_y = y[:, None] - gt_center_y[None,:]
    gs_r = short_size[None,:] - biu[:,None]
    gauss_D = np.exp(-((gs_x) ** 2 + (gs_y) ** 2) / (2 * gs_r * gs_r + 1e-5))
    # print(gauss_D.shape)

    l_off = x[:, None] - gt_boxes[..., 0][None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
    t_off = y[:, None] - gt_boxes[..., 1][None, :]
    r_off = gt_boxes[..., 2][None, :] - x[:, None]
    b_off = gt_boxes[..., 3][None, :] - y[:, None]
    biu_off = np.stack([l_off, t_off, r_off, b_off, targets_dt], axis=-1)  # [batch_size,h*w,m,4]
    # print(biu_off.shape)
    # areas = (biu_off[..., 0] + biu_off[..., 2]) * (biu_off[..., 1] + biu_off[..., 3])  # [batch_size,h*w,m]

    # radiu = stride
    #
    # center_off = np.sqrt(
    #     (x[:, None] - gt_center_x[None, :]) ** 2 + (y[:, None] - gt_center_y[None, :]) ** 2)
    # mask_center = center_off < radiu
    mask_center = gauss_D > gauss_range
    mask_pos = mig & mask_center  # [batch_size,h*w,m]

    # print(areas.shape)
    # print(gauss_D.shape)
    gauss_D[~mask_pos] = 0
    # print(gauss_D[mask_pos])
    # areas[~mask_pos] = 999999

    areas_min_ind = np.argmax(gauss_D,axis=-1)
    # print(areas_min_ind.shape)
    # m = areas.shape[-1]

    m = gauss_D.shape[-1]

    reg_targets = biu_off[np.eye(m)[areas_min_ind].astype(np.bool)]
    # print(reg_targets.shape)
    reg_targets[:, 0:4] = reg_targets[:, 0:4] / stride

    cls_targets = classes[np.eye(m)[areas_min_ind].astype(np.bool)].reshape(-1,1)
    target_pos = (np.arange(1, DefaultConfig.class_num + 1)[None, :] == cls_targets).astype(np.float32)

    tc_targets = theta_cls[np.eye(m)[areas_min_ind].astype(np.bool)].reshape(-1,1)
    tc_targets = (np.arange(1, 18 + 1)[None, :] == tc_targets).astype(np.float32)
    # print(tc_targets.shape)

    tr_targets = theta_reg[np.eye(m)[areas_min_ind].astype(np.bool)]
    # print(tr_targets.shape)


    mask_smooth = gauss_D[np.eye(m)[areas_min_ind].astype(np.bool)]
    mask_smooth = mask_smooth[:,np.newaxis]

    cnt_targets = np.zeros_like(cls_targets)
    mask_pos_2 = np.sum(mask_pos.astype(np.int32),axis=-1)
    mask_pos_2 = mask_pos_2 >= 1

    cls_targets[~mask_pos_2] = 0  # [batch_size,h*w,1]
    # print(cls_targets[mask_pos_2])
    reg_targets[~mask_pos_2] = -1
    #
    # print(reg_targets[mask_pos_2])
    # print(coords[mask_pos_2])
    # for i in range(0,len(reg_targets[mask_pos_2])):
    #     re = reg_targets[mask_pos_2][i]
    #     co = coords[mask_pos_2][i]
    cnt_targets[mask_pos_2] = 1
    # print(mask_pos_2.sum())
    target_pos[~mask_pos_2] = 0

    tc_targets[~mask_pos_2] = 0
    tr_targets[~mask_pos_2] = -1
    # smooth label
    # target_pos = target_pos * mask_smooth



    # return cls_targets.astype(np.float32), cnt_targets.astype(np.float32), reg_targets.astype(np.float32)
    return target_pos, cnt_targets.astype(np.float32), reg_targets.astype(np.float32),tc_targets,tr_targets.astype(np.float32)





