import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from models.config import cfg
import numpy as np

# UV pooling for a single image
def one_UVPooling(x, U,V, fc1, fc2):

    center_u = U.data.cpu().numpy()
    center_v = V.data.cpu().numpy()
    
    for partind in range(len(center_u)):
        x1 = max(0,center_u[partind]-3)
        y1 = max(0,center_v[partind]-3)
        x2 = min(13,center_u[partind]+4)
        y2 = min(13,center_v[partind]+4)
        if x1==0:
            x2=6
        if y1==0:
            y2=6
        if x2 == 13:
            x1=7
        if y2==13:
            y1=7

        if x2-x1>6:
            x2=x1+6
        if y2-y1>6:
            y2=y1+6
        
        # x[512,14,14]
        part_feature = x[:,int(x1):int(x2+1), int(y1):int(y2+1)] #.contiguous().view(-1)
        
        part_feature = part_feature.contiguous().view(-1)
        part_second_branch = fc1(part_feature) # 512D

        if partind == 0:
            second_fc_output=part_second_branch
        else:
            second_fc_output = torch.cat((second_fc_output, part_second_branch), 0) # 22 * 512D
        
    one_iuv_out = second_fc_output.view(-1) #11264 D
    
    one_iuv_out = fc2(one_iuv_out) #4096D
    
    return one_iuv_out # 4096-D
                            
def UVPooling(x, U,V, fc1, fc2):
    # batch
    batch_size = len(U)
    for k in range(batch_size):
        one_iuv_out = one_UVPooling(x[k], U[k],V[k], fc1,fc2)
       
        if k==0:
            uv_out = one_iuv_out
        else:
            uv_out = torch.cat((uv_out, one_iuv_out))
        
    uv_out = uv_out.view((batch_size, 4096))

    return uv_out    
