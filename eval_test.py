from evalutate.eval import *

from points_to_rect.points_to_rect import points_to_rect
import numpy as np
# from dataloader.hrsc_dataset import HRSCDataset
from dataloader.dota_dataset import DOTADataset
from dataloader.rsdd_dataset import RSDDDataset
from dataloader.ucas_dataset import UCASDataset
# test_dataset = HRSCDataset(dataset=DefaultConfig.test_txt,augment=True)
# test_dataset = DOTADataset(dataset=DefaultConfig.test_txt,augment=True)
test_dataset = RSDDDataset(dataset=DefaultConfig.test_txt,augment=True)
# test_dataset = UCASDataset(dataset=DefaultConfig.test_txt,augment=True)
mAP = single_scale_detect_(model_pth='./checkpoint/dota_epoch33_loss.pth',ds=test_dataset,target_size=[512],m_scale=True)
#
