import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import numpy as np

from models.config import cfg
from ROIPool import one_ROIPooling, ROIPooling

from UVPool import one_UVPooling, UVPooling

def Pooling(x, u,v, fc1,fc2, first_branch_out):
    if len(x)>1:
        single = False
    else:
        single = True
        
    landmarks = None
    if landmarks:
        if single: # a single image(used for 'demo')
            roi_pool = one_ROIPooling(x, landmarks, fc1)
            both_branch = torch.cat((first_branch_out, roi_out.view(1,4096)),0).view(-1)

        else: # batch
            roi_pool = ROIPooling(x, landmarks, fc1) # n landmarks--n rois(512*6*6)
    else: # no roi
        roi_pool = None

     # third branch -- 3D pooling and fc layer
    if len(u)>0 and len(v)>0:
        if single:
            uv_pool = one_UVPooling(x, u,v, fc1,fc2)
        else:
            uv_pool = UVPooling(x, u,v, fc1,fc2)
    else:
        uv_pool = None
        
    if roi_pool and uv_pool:
        pool_out = roi_pool + uv_pool
    elif roi_pool:
        pool_out = roi_pool
    else:
        pool_out = uv_pool

    return pool_out


