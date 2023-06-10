from dataloader.dota_dataset import DOTADataset,rbox_corner
from tqdm import tqdm
import cv2
import torch
from model.Collater import Rescale,Compose,Reshape,Normailize
from model.srdf import SRDFDetector
import numpy as np
from evalutate.functi_utils import decode_prediction,non_maximum_suppression
import os
from evalutate.map import eval_mAP
import shutil
import codecs
from dataloader.bbox import quad_2_rbox,rbox_2_quad,sort_corners
import time
from model.config import DefaultConfig
from DOTA_devkit.ResultMerge_multi_process import ResultMerge
from DOTA_devkit.ResultMerge import mergebypoly

import shutil
def prepare_gt(ds,source_p):
    # print('prepare_gt')
    img_list = ds._load_image_id()
    data_set = ds.data_D
    if os.path.exists(os.path.join(source_p, 'evalutate/' + data_set + '/ground-truth')):
        shutil.rmtree(os.path.join(source_p, 'evalutate/' + data_set + '/ground-truth'))
        os.makedirs(os.path.join(source_p, 'evalutate/' + data_set + '/ground-truth'))
    else:
        # shutil.rmtree(os.path.join(source_path, 'evalutate/' + data_set + '/detection-results'))
        os.makedirs(os.path.join(source_p, 'evalutate/' + data_set + '/ground-truth'))


    source_path = DefaultConfig.gt_path
    gt_path = os.path.join(source_p,'evalutate/'+data_set+'/ground-truth')
    # print(img_list)
    for idx, im_name in enumerate((img_list)):
        gt_file = im_name + '.txt'
        source_ = os.path.join(source_path, gt_file)
        target_ = os.path.join(gt_path, gt_file)
        # shutil.copy(source_,target_)
        with open(source_, 'r') as F:
            a = F.readlines()
        for c in a:
            b = c.split(' ')

            line = b[-2] + ' ' + b[0] + ' ' + b[1] + ' ' + b[2] + ' ' + b[3] + ' ' + b[4] + ' ' + b[5] + ' ' + b[
                6] + ' ' + b[7] + '\n'
            # print(line)
            with open(target_, 'a') as FF:
                FF.writelines(line)
from nms_wrapper import nms

def im_detect(model, src, target_sizes, use_gpu=True, conf=None):
    if isinstance(target_sizes, int):
        target_sizes = [target_sizes]
    if len(target_sizes) == 1:
        return single_scale_detect(model, src, target_size=target_sizes[0], use_gpu=use_gpu, conf=conf)
    else:
        ms_dets = None
        for ind, scale in enumerate(target_sizes):
            cls_dets = single_scale_detect(model, src, target_size=scale, use_gpu=use_gpu, conf=conf)
            if cls_dets.shape[0] == 0:
                continue
            if ms_dets is None:
                ms_dets = cls_dets
            else:
                ms_dets = np.vstack((ms_dets, cls_dets))
        if ms_dets is None:
            return np.zeros((0, 7))
        # cls_dets = np.hstack((ms_dets[:, 2:7], ms_dets[:, 1][:, np.newaxis])).astype(np.float32, copy=False)
        cls_dets = np.concatenate([ms_dets[:, 2:7], ms_dets[:, 1][:, np.newaxis],ms_dets[:, 0][:, np.newaxis]],axis=-1).astype(np.float32, copy=False)
        keep = nms(cls_dets, 0.1)
        return ms_dets[keep, :]
def single_scale_detect(model, src, target_size, use_gpu=True, conf=None):
    im, im_scales = Rescale(target_size=target_size, keep_ratio=True)(src)
    # print(im_scales)
    im = Compose([Normailize(), Reshape(unsqueeze=True)])(im)
    if use_gpu and torch.cuda.is_available():
        model, im = model.cuda(), im.cuda()
    with torch.no_grad():
        scores, classes, boxes = model(im)

    boxes[:, :4] = boxes[:, :4] / im_scales
    if boxes.shape[1] > 5:
        boxes[:, 5:9] = boxes[:, 5:9] / im_scales
    scores = np.reshape(scores, (-1, 1))
    classes = np.reshape(classes, (-1, 1))
    cls_dets = np.concatenate([classes, scores, boxes], axis=1)
    keep = np.where(classes > 0)[0]

    return cls_dets[keep, :]
    # cls, score, x,y,x,y,a,   a_x,a_y,a_x,a_y,a_a
def single_scale_detect_(model_pth,ds, target_size,m_scale=False, use_gpu=True):
    # source_path = 'C:/Users/60590/Desktop/SRDF_FPN/'
    # ds = DOTADataset(dataset=DefaultConfig.test_txt, augment=True)
    data_set = ds.data_D
    source_path = DefaultConfig.source_path
    if os.path.exists(os.path.join(source_path,'evalutate/'+data_set+'/detection-results')):
        shutil.rmtree(os.path.join(source_path, 'evalutate/' + data_set + '/detection-results'))
        os.makedirs(os.path.join(source_path, 'evalutate/' + data_set + '/detection-results'))
    else:
        # shutil.rmtree(os.path.join(source_path, 'evalutate/' + data_set + '/detection-results'))
        os.makedirs(os.path.join(source_path, 'evalutate/' + data_set + '/detection-results'))




    # results = {cat: {img_id: [] for img_id in ds._load_image_id()} for cat in ds.classes}
    prepare_gt(ds,source_path)
    model = SRDFDetector(mode="inference")
    model.load_state_dict(torch.load(model_pth))
    # model = model.cuda().eval()
    model.eval()
    # start = time.time()
    img_list = ds._load_image_names()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    for idx, im_name in enumerate(tqdm(img_list, desc=s)):
    # for idx, im_name in enumerate(tqdm(img_list)):

        src = cv2.cvtColor(cv2.imread(im_name, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # im, im_scales = Rescale(target_size=target_size, keep_ratio=True)(src)
        # # print(im_scales)
        # im = Compose([Normailize(), Reshape(unsqueeze=True)])(im)
        img_id = ds._load_image_id()[idx]

        # if use_gpu and torch.cuda.is_available():
        #     model, im = model.cuda(), im.cuda()
        # with torch.no_grad():
        #     scores, classes, boxes = model(im)
        #
        # boxes[:, :4] = boxes[:, :4] / im_scales
        # if boxes.shape[1] > 5:
        #     boxes[:, 5:9] = boxes[:, 5:9] / im_scales
        # scores = np.reshape(scores, (-1, 1))
        # classes = np.reshape(classes, (-1, 1))
        # cls_dets = np.concatenate([classes, scores, boxes], axis=1)
        # keep = np.where(classes > 0)[0]
        # dets =  cls_dets[keep, :]
        if m_scale:
            # scales = target_size[0] + 32 * np.array([x for x in range(-1, 5)])
            scales = [target_size[0]*0.5,target_size[0],target_size[0]*1.5]
        else:
            scales = target_size
        dets = im_detect(model, src, scales, True)


        nt += len(dets)
        # os.path.join('./evalutate/dota/detection-results', '{}.txt'.format(img_id)), 'a'
        out_file = os.path.join(os.path.join(os.path.join(source_path,'evalutate/'+data_set+'/detection-results'), '{}.txt'.format(img_id)))
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue
            # print(rbox_corner(dets[:, 2:]).shape)
            res = sort_corners(rbox_corner(dets[:, 2:],train=False))
            for k in range(dets.shape[0]):
                f.write('{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                    ds.return_class(dets[k, 0]), dets[k, 1],
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7])
                )
        assert len(os.listdir(os.path.join(os.path.join(source_path,'evalutate/'+data_set+'/'), 'ground-truth'))) != 0, 'No labels found in test/ground-truth!! '
        # print(out)

    mAP,All_ap = eval_mAP(os.path.join(source_path,'evalutate/'+data_set+'/'), use_07_metric=False)
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', len(img_list), nt, 0, 0, mAP, 0))
    return mAP,All_ap
def dota_evaluate(model_pth,
                  target_size,
                  test_path,m_scale= True,
                  conf = None):

    root_dir = os.path.join(DefaultConfig.source_path,'merge','outputs')
    res_dir = os.path.join(root_dir, 'detections')          # 裁剪图像的检测结果
    integrated_dir = os.path.join(root_dir, 'integrated')   # 将裁剪图像整合后成15个txt的结果
    merged_dir = os.path.join(root_dir, 'merged')           # 将整合后的结果NMS

    if  os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    for f in [res_dir, integrated_dir, merged_dir]:
        if os.path.exists(f):
            shutil.rmtree(f)
        os.makedirs(f)

    ds = DOTADataset(dataset=DefaultConfig.test_txt, augment=True)

    model = SRDFDetector(mode="inference")

    model.load_state_dict(torch.load(model_pth))
    # model = model.cuda().eval()
    model.eval()
    # start = time.time()

    ims_list = [x for x in os.listdir(test_path)]
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    for idx, im_name in enumerate(tqdm(ims_list, desc=s)):
        im_path = os.path.join(test_path, im_name)
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if m_scale:
            # scales = target_size[0] + 32 * np.array([x for x in range(-1, 5)])
            scales = [0.5 * target_size[0],0.75 * target_size[0], target_size[0],1.25 * target_size[0], target_size[0] * 1.5]
        else:
            scales = target_size
        dets = im_detect(model, im, target_sizes=scales, conf = conf)

        nt += len(dets)
        out_file = os.path.join(res_dir,  im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue

            res = sort_corners(rbox_corner(dets[:, 2:],train=False))
            for k in range(dets.shape[0]):
                f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {} {} {:.2f}\n'.format(
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7],
                    ds.return_class(dets[k, 0]), im_name[:-4], dets[k, 1],)
                )
    ResultMerge(res_dir, integrated_dir, merged_dir)
    ## calc mAP
    mergebypoly(integrated_dir, merged_dir)
    # # display result
    # pf = '%20s' + '%10.3g' * 6  # print format
    # print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0))
    # return 0, 0, mAP, 0
if __name__ == '__main__':
    dota_evaluate(model_pth='C:/Users/60590/Desktop/SRDF/checkpoint/dota_epoch58_loss.pth',
                  test_path=DefaultConfig.test_txt, target_size=[1024],conf=.18)
