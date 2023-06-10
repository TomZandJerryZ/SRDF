import torch
import numpy as np
import numpy.random as npr
import cv2
from torchvision.transforms import Compose
import torchvision.transforms as transforms
def rescale(im, target_size, max_size, keep_ratio, multiple=32):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if keep_ratio:
        # method1
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im_scale_x = np.floor(im.shape[1] * im_scale / multiple) * multiple / im.shape[1]
        im_scale_y = np.floor(im.shape[0] * im_scale / multiple) * multiple / im.shape[0]
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
        # method2
        # im_scale = float(target_size) / float(im_size_max)
        # im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # im_scale = np.array([im_scale, im_scale, im_scale, im_scale])

    else:
        target_size = int(np.floor(float(target_size) / multiple) * multiple)
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
    return im, im_scale
class Rescale(object):
    def __init__(self, target_size=600, max_size=2000, keep_ratio=True):
        self._target_size = target_size
        self._max_size = max_size
        self._keep_ratio = keep_ratio

    def __call__(self, im):
        if isinstance(self._target_size, list):
            random_scale_inds = npr.randint(0, high=len(self._target_size))
            target_size = self._target_size[random_scale_inds]
        else:
            target_size = self._target_size
        im, im_scales = rescale(im, target_size, self._max_size, self._keep_ratio)
        return im, im_scales


class Normailize(object):
    def __init__(self):
        # RGB: https://github.com/pytorch/vision/issues/223
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 均值和方差
        ])

    def __call__(self, im):
        im = self._transform(im)
        return im


class Reshape(object):
    def __init__(self, unsqueeze=True):
        self._unsqueeze = unsqueeze
        return

    def __call__(self, ims):
        if not torch.is_tensor(ims):
            ims = torch.from_numpy(ims.transpose((2, 0, 1)))
        if self._unsqueeze:
            ims = ims.unsqueeze(0)
        return ims
# TODO: keep_ratio

class Collater(object):
    """"""
    def __init__(self, scales,train_size, keep_ratio=False, multiple=32):
        if isinstance(scales, (int, float)):
            self.scales = np.array([scales], dtype=np.int32)
        else:
            self.scales = np.array(scales, dtype=np.int32)
        self.keep_ratio = keep_ratio
        self.multiple = multiple
        self.train_size = train_size


    # def __int__(self):
    def __call__(self, batch):
        biu = npr.randint(0, 10)
        if biu < 5:
            random_scale_inds = npr.randint(0, high=len(self.scales))
            target_size = self.scales[random_scale_inds]
        else:
            target_size = self.train_size

        # random_scale_inds = npr.randint(0, high=len(self.scales))
        # target_size = self.scales[random_scale_inds]
        target_size = int(np.floor(float(target_size) / self.multiple) * self.multiple)
        rescale = Rescale(target_size=target_size, keep_ratio=self.keep_ratio)
        transform = Compose([Normailize(), Reshape(unsqueeze=False)])

        images = [sample['image'] for sample in batch]
        bboxes = [sample['boxes'] for sample in batch]
        # wht = [sample['wht'] for sample in batch]
        # cls = [sample['cls'] for sample in batch]
        batch_size = len(images)
        max_width, max_height = -1, -1
        for i in range(batch_size):
            im, _ = rescale(images[i])
            height, width = im.shape[0], im.shape[1]
            max_width = width if width > max_width else max_width
            max_height = height if height > max_height else max_height

        padded_ims = torch.zeros(batch_size, 3, max_height, max_width)

        num_params = bboxes[0].shape[-1]
        max_num_boxes = max(bbox.shape[0] for bbox in bboxes)
        padded_boxes = torch.ones(batch_size, max_num_boxes, num_params) * -1
        # padded_wht = torch.ones(batch_size, max_num_boxes, 3) * -1
        # padded_cls = torch.ones(batch_size, max_num_boxes, 1) * 0
        for i in range(batch_size):
            im, bbox = images[i], bboxes[i]
            im, im_scale = rescale(im)
            height, width = im.shape[0], im.shape[1]
            padded_ims[i, :, :height, :width] = transform(im)
            if num_params < 9:
                bbox[:, :4] = bbox[:, :4] * im_scale
            else:
                bbox[:, :8] = bbox[:, :8] * np.hstack((im_scale, im_scale))

            cl = bbox[:,-1]


            padded_boxes[i, :bbox.shape[0], :] = torch.from_numpy(bbox)

        return {'image': padded_ims, 'boxes': padded_boxes[:,:,:-1] ,'classes':padded_boxes[:,:,-1]}