import os
import random

import cv2
import sys
import numpy as np
import torch
import torch.utils.data as data

from dataloader.imaugments import *
# from utils.utils import plot_gt
from dataloader.bbox import quad_2_rbox,rbox_2_quad,sort_corners,quad_2_rbox_180
from evalutate.calculate_F import *
from model.config import DefaultConfig

class HRSCDataset(data.Dataset):

    def __init__(self,
                 dataset=None,
                 augment=False,
                 level=1,
                 only_latin=True):
        self.level = level
        self.image_set_path = dataset
        if dataset is not None:
            self.image_list = self._load_image_names()
        if self.level == 1:
            self.classes = ('__background__',  'ship')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.augment = augment
        self.n = len(self.image_list)
        self.indices = range(self.n)
        self.labels = self.labels()
        self.data_D = 'HRSC'

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        index = self.indices[index]
        im_path = self.image_list[index]
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        roidb = self._load_annotation(self.image_list[index])
        gt_inds = np.where(roidb['gt_classess'] != 0)[0]
        # print(roidb['gt_classess'].shape)
        bboxes = roidb['boxess'][gt_inds, :]
        classes = roidb['gt_classess'][gt_inds]
        gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)

        transform = Augment([HSV(0.5, - 0.5, p=0.5),  # Augment中多有的p代表概率 0.5 假设100个样本 会有44-50个样本进行改变
                             HorizontalFlip(p=0.5),
                             VerticalFlip(p=0.5),
                             Affine(degree=0, translate=0.1, scale=0.2, p=0.5),#会出现问题 很多标签 位移或者形变
                             Noise(0.01, p=0.3),
                             Blur(1.3, p=0.5),
                             ], box_mode='xyxyxyxy')
        im, bbx = transform(im, bboxes)
        img, bbx = rotate_img_bbox(im, bbx)
        bbx = bbx.reshape(bbx.shape[0], 8)

        mask = mask_valid_boxes(quad_2_rbox(bbx, 'xywha'), return_mask=True)
        bbx = bbx[mask]
        gt_boxes = gt_boxes[mask]
        classes = classes[mask]

        bbx = sort_corners(bbx)
        rect = quad_2_rbox_180(bbx)
        angle = Angle_0_180(bbx)
        rect[:, -1] = angle[:, 0]

        for i, bbox in enumerate(rect):
            # gt_boxes[i, :5] = quad_2_rbox(np.array(bbox), mode='xyxya')
            # gt_boxes[i,4] = 45 - gt_boxes[i,4]
            gt_boxes[i, :5] = rect[i]

            gt_boxes[i, 5] = classes[i]


        return {'image': img, 'boxes': gt_boxes, 'path': im_path}

    def labels(self):
        labels = []
        for i in range(0, self.n):
            index = self.image_list[i]
            root_dir = index.split(r'images')[0]
            label_dir = os.path.join(root_dir, 'labelTxt')
            _, img_name = os.path.split(index)

            filename = os.path.join(label_dir, img_name[:-4] + '.txt')
            label = []
            with open(filename, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                objects = content.split('\n')

                for obj in objects:
                    if len(obj) != 0:
                        *box, class_name, difficult = obj.split(' ')
                        if difficult == 2:
                            continue
                        box = [eval(x) for x in obj.split(' ')[:8]]
                        gt_cls = self.class_to_ind[class_name]-1
                        box.append(gt_cls)
                    label.append(box)
            labels.append(label)
        return labels


    def _load_image_names(self):
        """
        Load the names listed in this dataset's image set file.
        """
        image_set_file = self.image_set_path
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            # image_list = [x.strip() for x in f.readlines()]
            image_list = [os.path.join(r'C:\Users\60590\Desktop\HRSC\images',x.strip()+'.png',) for x in f.readlines()]
            # r'C:\Users\sparrow\Desktop\dota_test\images'
            # C:\Users\savvy\Desktop\dota
        # print(image_list)
        return image_list

    def _load_image_id(self):
        """
        Load the names listed in this dataset's image set file.
        """
        image_set_file = self.image_set_path
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_list = [x.strip() for x in f.readlines()]
            # image_list = [os.path.join(r'C:\\Users\\savvy\\Desktop\\dota\\images\\',x.strip()+'.png',) for x in f.readlines()]
            # r'C:\Users\sparrow\Desktop\dota_test\images'
            # C:\Users\savvy\Desktop\dota
        # print(image_list)
        return image_list

    def _load_annotation(self, index):
        # print(index)
        root_dir = index.split(r'images')[0]
        label_dir = os.path.join(root_dir, 'labelTxt')
        _, img_name = os.path.split(index)

        filename = os.path.join(label_dir, img_name[:-4] + '.txt')
        boxes, gt_classes = [], []
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            objects = content.split('\n')
            for obj in objects:
                if len(obj) != 0:
                    *box, class_name, difficult = obj.split(' ')
                    if difficult == 2:
                        continue
                    box = [eval(x) for x in obj.split(' ')[:8]]
                    label = self.class_to_ind[class_name]
                    boxes.append(box)
                    gt_classes.append(label)
        return {'boxess': np.array(boxes, dtype=np.float32), 'gt_classess': np.array(gt_classes)}

    def display(self, boxes, img_path):
        img = cv2.imread(img_path)
        for box in boxes:
            coors = box.reshape(4, 2).astype(np.int32)
            img = cv2.polylines(img, [coors], True, (0, 0, 255), 2)
        cv2.imshow(img_path, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def return_class(self, id):
        id = int(id)
        return self.classes[id]

    # 统计类别数目
    def sta_cls(self):
        dict_cls = {}
        for i in self.classes:
            if i != '__background__':
                dict_cls[i] = 0
        for i in range(len(self.indices)):
            index = self.indices[i]
            im_path = self.image_list[index]
            # im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            roidb = self._load_annotation(self.image_list[index])
            gt_inds = np.where(roidb['gt_classess'] != 0)[0]

            classes = roidb['gt_classess'][gt_inds]
            for c in classes:
                dict_cls[self.classes[c]] = dict_cls[self.classes[c]] + 1



        return dict_cls
    def test_crop(self):

        for i in range(0,self.__len__()):
            i = self.indices[
                i
            ]
            im_path = self.image_list[i]
            imgg = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            print(self.image_list[i])
            anno = self._load_annotation(self.image_list[i])

            bbx_nt = anno['boxess']
            start = {}
            cls_nt = anno['gt_classess']

            start['boxes'] = bbx_nt
            start['gt_classes'] = cls_nt
            transform = Augment([HSV(0.5, - 0.5, p=0.5),  # Augment中多有的p代表概率 0.5 假设100个样本 会有44-50个样本进行改变
                                 HorizontalFlip(p=0.5),
                                 VerticalFlip(p=0.5),
                                 Affine(degree=0, translate=0.1, scale=0.2, p=0.5),#会出现问题 很多标签 位移或者形变
                                 Noise(0.01, p=0.3),
                                 Blur(1.3, p=0.5),
                                 ], box_mode='xyxyxyxy')
            img, bbx = transform(imgg, start['boxes'])
            img,bbx = rotate_img_bbox(img,bbx)
            bbx = bbx.reshape(bbx.shape[0],8)
            mask = mask_valid_boxes(quad_2_rbox(bbx.astype(np.int32), 'xywha'), return_mask=True)
            bbx = bbx[mask]
            # gt_boxes = gt_boxes[mask]
            # classes = classes[mask]
            # print(bbx.shape)
            bbx = sort_corners(bbx)


            rect = quad_2_rbox_180(bbx)

            angle = Angle_0_180(bbx)
            rect[:,-1] = angle[:,0]
            IBO = IBO_to_cv2(rect)
            corn = rbox_corner(rect)
            img = img.copy()

            for i,box in enumerate(corn):

                coors = box.reshape(4, 2).astype(np.int32)
                coors3 = IBO[i].reshape(4, 2).astype(np.int32)
                coor2 = corn[i,:].reshape(4, 2).astype(np.int32)
                x1,y1,x2,y2 = rect[i,:4].astype(np.int32)
                area = (x2-x1)*(y2-y1)
                c = (int(x1),int(y1))

                cv2.rectangle(img,(x1,y1),(x2,y2 ), (0,255,0), 2)
                cv2.polylines(img, [coor2], True, (255, 0, 0), 3)
                cv2.polylines(img, [coors3], True, (255, 0, 255), 4)
                im = cv2.polylines(img, [coors], True, (0, 0, 255), 2)
                cv2.circle(img, (coors[0, 0], coors[0, 1]), 4, (255, 255, 255), 3)
                cv2.putText(img, str(area), (coors[0, 0], coors[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
            cv2.imshow('img_path', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def IBO_to_cv2(rect):
    num_obj = rect.shape[0]
    boxes = np.zeros((num_obj, 8))
    for i in range( num_obj):
        x0 ,y0,x1,y1,angle = rect[i]
        if angle > 90:
            angle = angle - 180
        else:
            angle = angle
        angle = -angle
        x,y = (x0+x1)/2,(y0+y1)/2
        w,h = x1-x0,y1-y0
        boxes[i] = cv2.boxPoints(((x,y),(w,h),angle)).reshape(8,)

    return boxes



def rotate_img_bbox(img,bbx,p=0.5):
    # nAgree = angle
    h,w = img.shape[:2]
    bbx = bbx.reshape(bbx.shape[0],4,2)
    if np.random.rand(1) < p:
        scale = 1
        random_degree = np.random.rand(1)[0] * 90
        dRot = random_degree * np.pi / 180

        nw = (abs(np.sin(dRot) * h) + abs(np.cos(dRot) * w)) * scale
        nh = (abs(np.cos(dRot) * h) + abs(np.sin(dRot) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), random_degree, scale)

        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rotat_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        label = []
        for i in range(0, bbx.shape[0]):
            r_bbx = []
            for j in bbx[i]:
                point = np.dot(rot_mat, np.array([j[0], j[1], 1]))
                r_bbx.append(point.tolist())
            label.append(r_bbx)

        label = np.stack(label, axis=0)
    else:
        rotat_img = img.copy()
        label = np.copy(bbx)



    return rotat_img , label.astype(np.int32)



def dot_product_angle(v1, v2):

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        print("Zero magnitude vector!")
    else:
        vector_dot_product = np.dot(v1, v2)
        arccos_ang = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = np.degrees(arccos_ang)
        # angle = arccos_ang*180/np.pi
        return angle
    return 0



def Angle_0_180(bbx):
    num_obj = bbx.shape[0]
    x_asix = np.array([2, 0])
    x_asix = np.expand_dims(x_asix, 0).repeat(num_obj, axis=0)
    A_test = np.array(bbx).reshape(num_obj, 4, 2)
    A_test[:, :, -1] = -A_test[:, :, -1]
    start_points = A_test[:, 0, :]
    cand_point1 = A_test[:, 1, :]
    cand_point2 = A_test[:, -1, :]

    a = np.square((start_points - cand_point1)[:, 0]) + np.square((start_points - cand_point1)[:, 1])
    b = np.square((start_points - cand_point2)[:, 0]) + np.square((start_points - cand_point2)[:, 1])

    mask = (a - b) > 0
    V_ = np.zeros((num_obj, 2))
    V_[mask] = A_test[mask][:, 0, :] - A_test[mask][:, 1, :]
    if A_test[~mask].shape[0] != 0:
        V_[~mask] = A_test[~mask][:, 0, :] - A_test[~mask][:, -1, :]
    angles = []
    # print(V_)
    m2 = V_[:,-1] <0
    V_[m2] = -V_[m2]
    for i in range(num_obj):
        angle = (dot_product_angle(V_[i], x_asix[i]))
        if angle >= 180:
            angle = 0
        angles.append(round(angle,3))
    angles = np.array(angles).reshape(num_obj, 1)
    return angles

def rbox_corner(rbbx,train=True):
    corners = []
    num_bbx = rbbx.shape[0]
    for i,quad in enumerate(rbbx):
        xy = quad[:4]
        angle = quad[-1]
        c_x,c_y = (xy[0]+xy[2])/2,(xy[1]+xy[3])/2
        rects = np.array([[xy[0],xy[1]],[xy[2],xy[1]],[xy[2],xy[3]],[xy[0],xy[3]]]) - np.array([[c_x,c_y]])
        if angle > 90:
            angle = angle - 180
        else:
            angle = angle
        theta = math.radians(angle)
        matrix_left = np.array([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)]).reshape(2,
                                                                                                              2)  # 逆时针
        rect = rects.dot(matrix_left) + np.array([[c_x,c_y]])
        corners.append(rect)

    if train:
        corners = np.array(corners).reshape(num_bbx, 4, 2)
    else:
        corners = np.array(corners).reshape(num_bbx, 8)

    return corners



def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    # Usage: index = random.choices(range(n), weights=image_weights, k=1)  # weighted image sample
    class_counts = np.array([np.bincount(x[:, 0].astype(int), minlength=nc) for x in labels])
    return (class_weights.reshape(1, nc) * class_counts).sum(1)
if __name__ == '__main__':

    ds = HRSCDataset(dataset=r'C:\Users\60590\Desktop\HRSC\test.txt',augment=True)
    # labels = ds.labels
    # # for x in labels:
    # #     print(x[:,-1])
    # labels = np.concatenate(labels, 0)
    # classes = labels[:, -1].astype(int)  # labels = [class xywh]
    #
    # weights = np.bincount(classes, minlength=(DefaultConfig.class_num))  # occurrences per class
    # weights[weights == 0] = 1  # replace empty bins with 1
    # weights = 1 / weights  # number of targets per class
    # weights /= weights.sum()  # normalize
    # w = torch.from_numpy(weights).float()
    # maps = np.zeros(DefaultConfig.class_num)
    # cw = w * (1 - maps) ** 2 / (DefaultConfig.class_num)
    # # print(cw)
    #
    # ll = [np.array(x) for x in ds.labels]
    # # for x in ll:
    # #     print(x.shape)
    # class_counts = np.array([np.bincount(x[:, -1].astype(int), minlength=15) for x in ll])
    # iw = (cw.reshape(1, DefaultConfig.class_num) * class_counts).sum(1)
    #
    # ds.indices = random.choices(range(ds.n), weights=iw, k=ds.n)
    # print(ds.indices)
    ds.test_crop()
    # print(ds.sta_cls())










