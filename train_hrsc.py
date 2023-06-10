import os
import numpy as np
from model.srdf import SRDFDetector
import torch
# from dataloader.dota_dataset import DOTADataset
from dataloader.hrsc_dataset import HRSCDataset
import math, time
from model.Collater import Collater
mixed_precision = False
from evalutate.eval import *
from model.config import DefaultConfig
from sample_balance import *
import random

from torch.cuda import amp
re_train=False
# torch.manual_seed(3407)
train_dataset = HRSCDataset(dataset=DefaultConfig.train_txt,augment=True)
test_dataset = HRSCDataset(dataset=DefaultConfig.test_txt,augment=True)
class_weights = labels_to_class_weights(train_dataset.labels, DefaultConfig.class_num) * DefaultConfig.class_num
maps = np.zeros(DefaultConfig.class_num)



model = SRDFDetector(mode="training").cuda()
# 在已有模型的基础上继续训练.3
if re_train:
    start_epoch = 12
    model_p = "./checkpoint/dota_epoch%d_loss.pth" % (start_epoch)
    model.load_state_dict(torch.load(model_p))
else:
    start_epoch = 0
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(model.parameters(),lr=8e-4,momentum=0.9,weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = 0.1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,3], gamma=0.5)
# if mixed_precision:
#     model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
BATCH_SIZE = 2 # 一次训练多少张图像
EPOCHS = 99 # 总训练轮数
WARMPUP_STEPS_RATIO = 0.12
multi_scale = True
training_size = 800
if multi_scale:
    scales = training_size + 32 * np.array([x for x in range(-1, 5)])
    # set manually
    # scales = np.array([384, 480, 544, 608, 704, 800, 896, 960])
    print('Using multi-scale %g - %g' % (scales[0], scales[-1]))
else:
    scales = training_size
collater = Collater(scales=scales,train_size=training_size, keep_ratio=True, multiple=32)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=collater)

steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = TOTAL_STEPS * WARMPUP_STEPS_RATIO

GLOBAL_STEPS = steps_per_epoch * start_epoch
LR_INIT = 5e-5
LR_END = 1e-6



# 余弦退火
def lr_func():
    if GLOBAL_STEPS < WARMPUP_STEPS:
        lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
    else:
        lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
            (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
        )
    return float(lr)

# model.half()
model.train()

aa = []
for epoch in range(start_epoch,EPOCHS):
    # print(maps)
    # cw = class_weights.cpu().numpy() * (1 - maps) ** 2 / DefaultConfig.class_num  # class weights
    # iw = labels_to_image_weights(train_dataset.labels, nc=DefaultConfig.class_num, class_weights=cw)  # image weights
    # train_dataset.indices = random.choices(range(train_dataset.n), weights=iw, k=train_dataset.n)

    # print(('\n' + '%10s' * 12) % ('Epoch', 'gpu_mem', 'cls', 'att', 'reg', 'r_r', 'P2_c', 'P2_r', 'P2_a' ,'total', 'targets', 'img_size'))
    # pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # progress bar
    print(('\n' + '%10s' * 14) % (
    'Epoch', 'gpu_mem', 'cls', 'att', 'reg', 'r_r', 'P2_c', 'P2_r', 'P2_a', 'total', 'targets', 'img_size', 'Lr', 'GS'))
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # progress bar
    sum_loss = 0
    for i, (epoch_step, data) in enumerate(pbar):

        batch_imgs, batch_boxes, batch_classes = data['image'],data['boxes'],data['classes']
        # for j in range(2):
        #     img = batch_imgs[j]
        #     img = img.cpu().numpy()
        #     # print(img.shape)
        #     cv2.imshow('img_path', img.reshape(800, 800, 3))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        '''lr warmup'''
        lr = lr_func()
        for param in optimizer.param_groups:
            param['lr'] = lr
            # lr=param['lr']


        with amp.autocast(enabled=mixed_precision):
            losses = model([batch_imgs, batch_boxes, batch_classes])

            cls_loss, cnt_loss, reg_loss,loss_refine_reg,P2_cls_loss , P2_reg_loss , P2_cnt_loss,total_loss = losses
            loss = total_loss

        loss.backward()
        if (i+1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
        s = ('%10s' * 2 + '%10.3g' * 12) % (
            '%g/%g' % (epoch, EPOCHS - 1), '%.3gG' % mem, cls_loss, cnt_loss, reg_loss, loss_refine_reg, P2_cls_loss,
            P2_reg_loss, P2_cnt_loss, total_loss, batch_boxes.shape[1], min(batch_imgs.shape[2:]), lr, GLOBAL_STEPS)

        sum_loss += total_loss
        pbar.set_description(s)
    # scheduler.step()
        GLOBAL_STEPS += 1
    maps = []
    lloss = sum_loss / len(train_loader)
    if epoch > -1:

       model_path = DefaultConfig.source_path+"checkpoint/dota_epoch%d_loss.pth" % (epoch + 1)
       torch.save(model.state_dict(), model_path)

       if os.path.exists(model_path):
           torch.cuda.empty_cache()
           mAP,All_AP = single_scale_detect_(model_pth=model_path,ds=test_dataset, target_size=[800])

           aa.append([epoch,float(lloss),mAP])

           with open('aa.txt','a')as F:
               F.writelines(str(aa)+'\n')
           for i in train_dataset.classes:
               if i != '__background__':
                   maps.append(All_AP[i])

    maps = np.array(maps)












