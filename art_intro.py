from evalutate.eval import *

import numpy as np
from model.gauss_p2 import *
from dataloader.dota_dataset import DOTADataset
from dataloader.ucas_dataset import UCASDataset
# test_dataset = HRSCDataset(dataset=DefaultConfig.test_txt,augment=True)
dataset = DOTADataset(dataset=DefaultConfig.train_txt,augment=False)
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
for i in range(98, dataset.__len__()):
    i = dataset.indices[
        i
    ]
    im_path = dataset.image_list[i]
    img = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    print(dataset.image_list[i])
    anno = dataset._load_annotation(dataset.image_list[i])
    length = DefaultConfig.limit_range
    bbx_nt = anno['boxess']
    cls_nt = anno['gt_classess']
    img, im_scales = Rescale(target_size=584, keep_ratio=True)(img)
    # print(bbx_nt.shape)
    # im = Compose([Normailize(), Reshape(unsqueeze=True)])(im)
    bbx_nt[:, :4] = bbx_nt[:, :4] * im_scales
    if bbx_nt.shape[1] > 5:
        bbx_nt[:, 4:] = bbx_nt[:, 4:] * im_scales
    cls_map_temp = np.zeros((576, 576), dtype=np.float32)
    for i, box in enumerate(bbx_nt):
        coors = box.reshape(4, 2).astype(np.int32)
        (x,y),(w,h),t = cv2.minAreaRect(coors)


        area = w*h
        fm =np.zeros((3,584//32,584//32))
        coords = np_coords_fmap2orig(fm,32)
        # for ii in coords:
        #     # print(ii[0])
        #     cv2.circle(img, (int(ii[0]), int(ii[1])), 1, (0, 255, 0), 2)
        # print(length[0][1])
        if area<length[0][1]:
            color = (0, 0, 255)
            radius = gaussian_radius((math.ceil(w), math.ceil(h)))
            radius = max(0, int(radius))
            cls_map_temp = draw_umich_gaussian(cls_map_temp, np.array([x, y]), radius)
            im = cv2.polylines(img, [coors], True, color, 2)
        # else:
        #     color = (0, 255, 0)
        #     im = cv2.polylines(img, [coors], True, color, 2)
        # radius = gaussian_radius((math.ceil(w), math.ceil(h)))
        # radius = max(0, int(radius))
        # cls_map_temp = draw_umich_gaussian(cls_map_temp, np.array([x, y]), radius)
        # cls_map_temp = cls_map_temp.astype(np.uint8)
        # cls_map_temp = cv2.applyColorMap(cls_map_temp,cv2.COLORMAP_JET)
        # im = cv2.polylines(img, [coors], True, color, 2)

        # im = cv2.addWeighted(cls_map_temp,.3,im,.7,0)
        # cv2.circle(img, (coors[0, 0], coors[0, 1]), 4, (255, 255, 255), 3)
        # cv2.putText(img, str(w*h), (coors[0, 0], coors[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
    cls_map_temp = np.uint8(255*cls_map_temp)
    cls_map_temp = cv2.applyColorMap(cls_map_temp, cv2.COLORMAP_JET)

    im = cv2.addWeighted(cls_map_temp,.7,im,.3,0)
    cv2.imshow('img_path', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()