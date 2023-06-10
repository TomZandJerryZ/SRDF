
import cv2
import math
import numpy as np
'''用于将回归特征图上的label转换为真实的bbx并查看'''
class Decoder_REG_Targets:
    def decoder_cf(self):
        return
    def decoder_reg(self,center,wh,angle,nine=False):
        '''part1.中心点，尺寸，角度[0-180)encoder的时候是根据最长边与x轴正向的夹角'''
        '''part2.中心点，尺寸，角度[0-90)encoder的时候是根据cv2.minAreaRect获取的'''
        if nine:
            cen_x,cen_y = center
            bbox_w, bbox_h = wh
            theta = angle
            bbx = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))
        else:
            h, w = wh
            cen_x, cen_y = center

            if angle > 90:
                angle = angle - 180
            else:
                angle = angle
            theta = math.radians(angle)

            # theta = 90 - theta
            # print(theta)
            bbx_x_asix = [[-h/2,w/2],[h/2,w/2],[h/2,-w/2],[-h/2,-w/2]]
            matrix_left = np.array([math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)]).reshape(2,2)  # 逆时针
            img_bbx = []
            for coor in bbx_x_asix:
                img_coor = np.matmul(matrix_left,np.array(coor).reshape(2,1))
                img_coor_x ,img_coor_y = cen_x + img_coor[0] ,cen_y - img_coor[1]
                img_bbx.append([img_coor_x ,img_coor_y])
            bbx = np.array(img_bbx).reshape(4,2)
            xyxy = np.array(bbx_x_asix)
            xyxy[:,0] = cen_x + xyxy[:,0]
            xyxy[:, 1] = cen_y - xyxy[:, 1]
            x1,y1 = np.min(xyxy[:,0]),np.min(xyxy[:,1])
            x2, y2 = np.max(xyxy[:, 0]), np.max(xyxy[:, 1])
            boxyxy = [x1,y1,x2, y2]
        return bbx,boxyxy


'''根据对角度进行处理，或将角度进行分类或者回归处理'''
class Theta_Switch:
    # def __init__(self):
        # self.bbx = bbx.reshape(4,2)
        # self.cv_theta = cv_theta

    def distance(self,P1,P2):
        '''计算两点间距离'''
        dis = np.sqrt(np.square(P1[0] - P2[0]) + np.square(P1[1] - P2[1]))
        return dis


    def Theta(self,box):
        box = box.reshape(4,2)
        results = {}
        c12, c23, c34, c14 = (box[0, :] + box[1, :]) / 2, (box[1, :] + box[2, :]) / 2, (box[2, :] + box[3, :]) / 2, (
                    box[0, :] + box[3, :]) / 2
        cen_x, cen_y = np.sum(box[:, 0]) / 4, np.sum(box[:, 1]) / 4
        ct = np.asarray([cen_x, cen_y], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        flos = ct - ct_int
        if self.distance(c12, c34) > self.distance(c23, c14):
            theta = self.vec_to_theta(c12, c34)
            bbox_w, bbox_h = self.distance(c12, c34), self.distance(c23, c14)  # w长边
            # return theta
        elif self.distance(c12, c34) == self.distance(c23, c14):
            bbox_w, bbox_h = self.distance(c12, c34), self.distance(c23, c14)
            theta1 = self.vec_to_theta(c12, c34)
            theta2 = self.vec_to_theta(c23, c14)
            theta = min(theta1, theta2)
        else:
            theta = self.vec_to_theta(c23, c14)
            bbox_w, bbox_h = self.distance(c23, c14), self.distance(c12, c34)  # w长边

        # results['center'] = [cen_x, cen_y]
        results['res'] = [cen_x, cen_y,bbox_w, bbox_h, theta]
        # results['theta'] = theta
        # results['offsets'] = flos

        return results

    def vec_to_theta(self,point1, point2):
        px1, py1 = point1
        px2, py2 = point2
        x2, y2 = 2, 0
        if px1 > px2:
            vx1, vy1 = px1 - px2, py1 - py2
        else:
            vx1, vy1 = px2 - px1, py2 - py1
        x1, y1 = vx1, -vy1

        cosa = (x1 * x2 + y1 * y2) / (math.sqrt(x1 * x1 + y1 * y1) * math.sqrt(x2 * x2 + y2 * y2))
        cosa = np.clip(np.array([cosa]),-1.0,1.0)[0]
        theta = math.degrees(math.acos(cosa))
        # print(theta)
        if y1 >= 0:
            theta = theta
        else:
            theta = 180 - theta
        if theta >= 179:
            theta = 0
        theta = int(theta*100) / 100
        # print(theta)
        return theta
