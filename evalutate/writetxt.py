#

import os
import cv2

# os.getcwd()
# imfile = [x for x in os.listdir('C:/Users/savvy/BBAVectors-Oriented-Object-Detection/Data/test/images')]
#
# for nam in imfile:
#     with open('test.txt','a')as file:
#         file.writelines(nam[:-4]+'\n')

im_p = r'C:\Users\savvy\Desktop\icdar2015\icdar2015\imgs\test'
label_p = r'C:\Users\savvy\Desktop\icdar2015\icdar2015\annotations\test'


# for i in os.listdir(im_p):
#     name = i.split('_')[0]
#     rest = i.split('_')[1]
#     lab_name = 'gt_'+i.split('.')[0]+'.txt'
#     # print(lab_name)
#     newname = name+'_1000'+rest
#     new_lb_name = 'gt_'+newname.split('.')[0]+'.txt'
#     # image = cv2.imread(os.path.join(im_p,i))
#     # cv2.imwrite(os.path.join(im_p,newname),image)
#     #
#     os.rename(os.path.join(im_p,i), os.path.join(im_p,newname))
#     os.rename(os.path.join(label_p,lab_name), os.path.join(label_p, new_lb_name))

# for i in os.listdir(im_p):
#     name = i.split('.')[0]
#     with open('icdar.txt','a')as F:
#         F.writelines(name+'\n')

import shutil,random
pp = r'C:\Users\savvy\Desktop\dota\images'

im_list = [x.split('.')[0] for x in os.listdir(pp)]

random.shuffle(im_list)

train = im_list[0:int(len(im_list)*.7)]
test = []
for i in im_list:
    if i not in train:
        test.append(i)
ppp = r'D:\srdf\Rotated-RetinaNet (2)\datasets\data_root\DOTA'
for i in train:
    with open(ppp+'\\train.txt', 'a') as F:
        F.writelines(os.path.join(pp,i+'.png')+'\n')

for i in test:
    with open(ppp+'\\test.txt', 'a') as F:
        F.writelines(os.path.join(pp,i+'.png')+'\n')