import os
from model.config import DefaultConfig
import cv2
color_pans = [(204,78,210),
                           (0,192,255),
                           (0,131,0),
                           (240,176,0),
                           (254,100,38),
                           (0,0,255),
                           (182,117,46),
                           (185,60,129),
                           (204,153,255),
                           (80,208,146),
                           (0,0,204),
                           (17,90,197),
                           (0,255,255),
                           (102,255,102),
                           (255,255,0)]
# from dataloader.dota_dataset import DOTADataset
from dataloader.hrsc_dataset import HRSCDataset
from dataloader.rsdd_dataset import RSDDDataset
# ds = DOTADataset(dataset=DefaultConfig.test_txt, augment=True)
# ds = HRSCDataset(dataset=DefaultConfig.test_txt, augment=True)
ds = RSDDDataset(dataset=DefaultConfig.test_txt, augment=True)
    # import shutil
import numpy as np
img_list = ds._load_image_id()
source_path = DefaultConfig.test_img_p

# gt_path = './evalutate/HRSC/detection-results'
gt_path = './evalutate/RSDD/detection-results'
save_p = r'C:\Users\60590\Desktop\RSDD_results'
# gt_path = './evalutate/dota/detection-results'
# print(img_list)
for idx, im_name in enumerate(img_list):
    im_p = os.path.join(source_path,im_name+'.jpg')
    im = cv2.imread(im_p)

    # cv2.imshow('>>',im)
    with open(os.path.join(gt_path,im_name+'.txt'),'r')as F:
        a = F.readlines()
    bbx = []
    cls = []
    score = []
    for i in a:
        cls.append(i.split('\n')[0].split(' ')[0])
        score.append(i.split('\n')[0].split(' ')[1])
        bbx.append(list(map(float,i.split('\n')[0].split(' ')[2:])))

    bbx = np.array(bbx)
    for i in range(0,len(bbx)):
        box = bbx[i]
        cl = cls[i]
        sc = score[i]
        coors = box.reshape(4, 2).astype(np.int32)
        cx,cy = int(np.sum(coors[:,0])/4),int(np.sum(coors[:,1])/4)
        # cv2.putText(im, str(sc)+' '+str(cl), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        im = cv2.polylines(im, [coors], True, color_pans[ds.classes.index(cl)-1], 2)
    if len(bbx)>0:
        # cv2.imwrite(os.path.join(save_p,im_name+'.jpg'),im)
        cv2.imshow('img_path', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



