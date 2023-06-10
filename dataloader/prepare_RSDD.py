import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import math
P = r'C:\Users\60590\Desktop\RSDD-SAR\RSDD-SAR\Annotations'
P2 = r'C:\Users\60590\Desktop\RSDD-SAR\RSDD-SAR\JPEGImages'

f = [x[:-4] for x in os.listdir(P)]

label_Txt = r'C:\Users\60590\Desktop\RSDD-SAR\RSDD-SAR\labelTxt'
if not os.path.exists(label_Txt):
    os.makedirs(label_Txt)

for i in f:
    path = os.path.join(P,i+'.xml')
    im_p = os.path.join(P2,i+'.jpg')
    label_p = os.path.join(label_Txt,i+'.txt')
    tree = ET.parse(path)
    root = tree.getroot()

    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    # print(width, height)

    im = cv2.imread(im_p)



    # <cx>299.8607</cx>
    #       <cy>83.7978</cy>
    #       <h>15.24122428894043</h>
    #       <w>5.936710834503174</w>
    #       <angle>-0.4691457200006939</angle>

    for obj in root.iter('object'):
        xml_box = obj.find('robndbox')
        cls = obj.find('name').text
        diff = obj.find('difficult').text
        cx = (float(xml_box.find('cx').text))
        cy = (float(xml_box.find('cy').text))
        h = (float(xml_box.find('h').text))
        w = (float(xml_box.find('w').text))
        angle = (float(xml_box.find('angle').text))

        # print(cx, cy, h, w, angle)
        angle = angle*180/math.pi
        bb = cv2.boxPoints(((cx, cy), (h, w), angle))
        # print(bb.reshape(8,))
        bb = bb.reshape(8,)
        lines = str(bb[0])+' '+str(bb[1])+' '+str(bb[2])+' '+str(bb[3])+' '+str(bb[4])+' '+str(bb[5])+' '+str(bb[6])+' '+str(bb[7])+' '+cls+' '+diff+'\n'

        with open(label_p,'a')as F:
            F.writelines(lines)
        # rect = cv2.minAreaRect(bb)
        # print(rect)
        # im = cv2.polylines(im, [bb.astype(np.int32)], True, (0, 0, 255), 2)
        # cv2.circle(img, (coors[0, 0], coors[0, 1]), 4, (255, 255, 255), 3)
        # cv2.putText(img, str(angle[i, 0]), (coors[0, 0], coors[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
    # cv2.imshow('img_path', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# path = r'C:\Users\60590\Desktop\RSDD-SAR\RSDD-SAR\Annotations\0_0_5.xml'
# im_p  =r'C:\Users\60590\Desktop\RSDD-SAR\RSDD-SAR\JPEGImages\0_0_5.jpg'

# tree = ET.parse(path)
# root = tree.getroot()
#
# size = root.find('size')
# width = float(size.find('width').text)
# height = float(size.find('height').text)
# print(width, height)
#
# im = cv2.imread(im_p)
#
# # <cx>299.8607</cx>
# #       <cy>83.7978</cy>
# #       <h>15.24122428894043</h>
# #       <w>5.936710834503174</w>
# #       <angle>-0.4691457200006939</angle>
#
# for obj in root.iter('object'):
#     xml_box = obj.find('robndbox')
#     cls = obj.find('name').text
#     diff = obj.find('difficult').text
#     cx = (float(xml_box.find('cx').text))
#     cy = (float(xml_box.find('cy').text))
#     h = (float(xml_box.find('h').text))
#     w = (float(xml_box.find('w').text))
#     angle = (float(xml_box.find('angle').text))
#
#     print(cx,cy,h,w,angle)
#     bb = cv2.boxPoints(((cx,cy),(h,w),angle))
#     im = cv2.polylines(im, [bb.astype(np.int32)], True, (0, 0, 255), 2)
#     # cv2.circle(img, (coors[0, 0], coors[0, 1]), 4, (255, 255, 255), 3)
#     # cv2.putText(img, str(angle[i, 0]), (coors[0, 0], coors[0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
# cv2.imshow('img_path', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
