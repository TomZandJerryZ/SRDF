import os
import torch
import numpy as np
import math
from DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast, py_cpu_nms_poly


def decoder_reg( center, wh, angle):
    '''part1.中心点，尺寸，角度[0-180)encoder的时候是根据最长边与x轴正向的夹角'''
    '''part2.中心点，尺寸，角度[0-90)encoder的时候是根据cv2.minAreaRect获取的'''

    h, w = wh
    cen_x, cen_y = center

    if angle > 90:
        angle = angle - 180
    else:
        angle = angle
    theta = math.radians(angle)

    # theta = 90 - theta
    # print(theta)
    bbx_x_asix = [[-h / 2, w / 2], [h / 2, w / 2], [h / 2, -w / 2], [-h / 2, -w / 2]]
    matrix_left = np.array([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)]).reshape(2,
                                                                                                          2)  # 逆时针
    img_bbx = []
    for coor in bbx_x_asix:
        img_coor = np.matmul(matrix_left, np.array(coor).reshape(2, 1))
        img_coor_x, img_coor_y = cen_x + img_coor[0], cen_y - img_coor[1]
        img_bbx.append([img_coor_x, img_coor_y])
    bbx = np.array(img_bbx).reshape(4, 2)

    return bbx

def decode_prediction(predictions, dsets,imscale):
    scores, classes, boxes, thetas = predictions
    # print(boxes.size)
    # print(boxes.shape)
    # assert len()
    pts0 = {cat: [] for cat in dsets.classes}
    scores0 = {cat: [] for cat in dsets.classes}
    if boxes.size == 0:
        return pts0, scores0
    else:
        for k, box in enumerate(boxes):
            # print(box.shape)
            w, h, cx, cy = box
            clse = classes[k]
            score = scores[k]
            theta = thetas[k]
            pts = decoder_reg((cx, cy), (w, h), theta)

            pts = pts.reshape(8,) / np.hstack((imscale, imscale))
            pts0[dsets.classes[int(clse)]].append(pts.reshape(4,2))
            scores0[dsets.classes[int(clse)]].append(score)
        return pts0, scores0



def non_maximum_suppression(pts, scores):
    nms_item = np.concatenate([pts[:, 0:1, 0],
                               pts[:, 0:1, 1],
                               pts[:, 1:2, 0],
                               pts[:, 1:2, 1],
                               pts[:, 2:3, 0],
                               pts[:, 2:3, 1],
                               pts[:, 3:4, 0],
                               pts[:, 3:4, 1],
                               scores[:, np.newaxis]], axis=1)
    nms_item = np.asarray(nms_item, np.float64)
    keep_index = py_cpu_nms_poly_fast(dets=nms_item, thresh=0.2)
    return nms_item[keep_index]


def write_results(
                  model,
                  dsets,
                  device,
                  decoder,
                  result_path,
                  print_ps=False):
    results = {cat: {img_id: [] for img_id in dsets.img_ids} for cat in dsets.category}
    for index in range(len(dsets)):
        data_dict = dsets.__getitem__(index)
        image = data_dict['image'].to(device)
        img_id = data_dict['img_id']
        image_w = data_dict['image_w']
        image_h = data_dict['image_h']

        with torch.no_grad():
            pr_decs = model(image)


        decoded_pts = []
        decoded_scores = []
        torch.cuda.synchronize(device)
        predictions = decoder.ctdet_decode(pr_decs)
        pts0, scores0 = decode_prediction(predictions, dsets)
        decoded_pts.append(pts0)
        decoded_scores.append(scores0)

        # nms
        for cat in dsets.category:
            if cat == 'background':
                continue
            pts_cat = []
            scores_cat = []
            for pts0, scores0 in zip(decoded_pts, decoded_scores):
                pts_cat.extend(pts0[cat])
                scores_cat.extend(scores0[cat])
            pts_cat = np.asarray(pts_cat, np.float32)
            scores_cat = np.asarray(scores_cat, np.float32)
            if pts_cat.shape[0]:
                nms_results = non_maximum_suppression(pts_cat, scores_cat)
                results[cat][img_id].extend(nms_results)
        if print_ps:
            print('testing {}/{} data {}'.format(index+1, len(dsets), img_id))

    for cat in dsets.category:
        if cat == 'background':
            continue
        with open(os.path.join(result_path, 'Task1_{}.txt'.format(cat)), 'w') as f:
            for img_id in results[cat]:
                for pt in results[cat][img_id]:
                    f.write('{} {:.12f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                        img_id, pt[8], pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))
