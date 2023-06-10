import torch
import numpy as np
import cv2
import math
from nms.cpu_nms import cpu_nms, cpu_soft_nms
from model.config import DefaultConfig
def points_to_rect(rbbx):

    num_bbx = rbbx.shape[0]
    boxes = np.zeros((num_bbx, 5))
    for i, quad in enumerate(rbbx):
        xy = quad[:4]
        angle = quad[-1]
        c_x, c_y = (xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2
        rects = np.array([[xy[0], xy[1]], [xy[2], xy[1]], [xy[2], xy[3]], [xy[0], xy[3]]]) - np.array([[c_x, c_y]])
        if angle > 90:
            angle = angle - 180
        else:
            angle = angle
        theta = math.radians(angle)
        matrix_left = np.array([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)]).reshape(2,
                                                                                                              2)  # 逆时针
        rect = rects.dot(matrix_left) + np.array([[c_x, c_y]])
        rect = rect.reshape(4,2)

        (x, y), (w, h), t = cv2.minAreaRect(rect.astype(np.int32))
        # bbx = cv2.boxPoints(rect).reshape(8,)
        x1, y1 = x - w * .5, y - h * .5
        x2, y2 = x + w * .5, y + h * .5
        boxes[i] = np.array([x1, y1, x2, y2, t])



    return boxes


#
# def nms(dets, thresh):
#     """Dispatch to either CPU or GPU NMS implementations."""
#     if dets.shape[0] == 0:
#         return []
#     if dets.shape[1] == 5:
#         raise NotImplementedError
#     elif dets.shape[1] == 6:
#         if torch.is_tensor(dets):
#             dets = dets.cpu().detach().numpy()
#             cls = dets[:, -1]
#             cls = cls[:, np.newaxis]
#             vector_box = dets[:, :-1]
#             rect = points_to_rect(vector_box).astype(np.float32)
#             detr = np.concatenate([rect, cls], axis=-1)
#         else:
#             cls = dets[:, -1]
#             cls = cls[:, np.newaxis]
#             vector_box = dets[:, :-1]
#             rect = points_to_rect(vector_box).astype(np.float32)
#             detr = np.concatenate([rect, cls], axis=-1)
#         return cpu_nms(detr, thresh)
#     else:
#         raise NotImplementedError

def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if dets.shape[1] == 5:
        raise NotImplementedError
    elif dets.shape[1] == 7:
        if torch.is_tensor(dets):
            dets = dets.cpu().detach().numpy()
            cls = dets[:, -2]
            id = dets[:,-1]
            cls = cls[:, np.newaxis]
            vector_box = dets[:, :-2]
            rect = points_to_rect(vector_box).astype(np.float32)
            detr = np.concatenate([rect, cls], axis=-1)
        else:
            cls = dets[:, -2]
            cls = cls[:, np.newaxis]
            id = dets[:, -1]
            vector_box = dets[:, :-2]
            rect = points_to_rect(vector_box).astype(np.float32)
            detr = np.concatenate([rect, cls], axis=-1)

        # A = np.zeros_like((id))
        # # print(detr.shape)
        # # A[[0,1,2,3,4]] = 1
        # # print(A)
        # for i in range(1,DefaultConfig.class_num+1):
        #
        #     if((np.where(id==i)[0].size)) > 0:
        #         # print(np.where(id == i))
        #         index = np.where(id==i)[0]
        #         # print(detr[index])
        #         index2 = cpu_nms(detr[index], thresh)
        #         B = np.zeros_like(A[index])
        #         B[index2] = 1
        #
        #         A[index] = B
        # # print(np.where(A>0)[0])
        # keep = np.where(A>0)[0]
        # # print(cpu_nms(detr, thresh))
        A_bsf = np.zeros_like((id))
        A_gtf = np.zeros_like((id))
        index_bsf = np.where(id != 14)[0]
        index_bsf_2 = cpu_nms(detr[index_bsf], thresh)
        B_bsf = np.zeros_like(A_bsf[index_bsf])
        B_bsf[index_bsf_2] = 1
        A_bsf[index_bsf] = B_bsf
        # bsf = 14
        # gtf = 7
        index_gtf = np.where(id != 7)[0]
        index_gtf_2 = cpu_nms(detr[index_gtf], thresh)
        B_gtf = np.zeros_like(A_bsf[index_gtf])
        B_gtf[index_gtf_2] = 1
        A_gtf[index_gtf] = B_gtf

        keep_bsf = np.where(A_bsf > 0)[0]
        keep_gtf = np.where(A_gtf > 0)[0]

        keep = keep_bsf.tolist() + keep_gtf.tolist()
        keep = list(set(keep))
        return keep
        # return cpu_nms(detr, thresh)
        # return keep.tolist()
    else:
        raise NotImplementedError