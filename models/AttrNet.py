import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from config import cfg
from layer_utils.Pool import Pooling
import numpy as np

class AttrNet(nn.Module):
    def __init__(self, num_classes=88, init_weights=True):
        super(AttrNet, self).__init__()
        
       # self.rois = rois
        # the first 4 shared conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # the first branch-- global image enter the 5th conv layer and fc1
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1,affine=True),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        self.fc2 = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        )
            
        self.second_branch_fc1 = nn.Sequential(
            nn.Linear(512*7*7, 512), #tbd
            nn.ReLU(True),
            nn.Dropout(),
        )

        self.second_branch_fc2 = nn.Sequential(
            nn.Linear(11264, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        
        self.fc3_fine_tune = nn.Linear(8192, cfg.num_classes)

    # forward model, if single = True, forward a single image
    # if single = False, forward batch
    def forward(self, x, u, v):
        # share first 4 conv layers
        x = self.conv1(x) # 112
        x = self.conv2(x) # 56
        x = self.conv3(x) # 28
        x = self.conv4(x) # 14
        
        # first branch-- continue to enter 5th conv layer

        first_branch_conv = self.conv5(x)
        
        first_branch_conv = first_branch_conv.view(first_branch_conv.size(0), -1)
        first_branch_out = self.fc1(first_branch_conv) # 4096D
        first_branch_out = self.fc2(first_branch_out) # 4096D

        pool_out = Pooling(x, u,v, self.second_branch_fc1, self.second_branch_fc2,first_branch_out) #4096D

        # concat the output from the first and the second branch
        both_branch = torch.cat((first_branch_out, pool_out), 1) # 8192D       
        
        output = self.fc3_fine_tune(both_branch)

        # for attribute prediction: return output
        # for image retrieval: return both_branch
        #return both_branch
        return output

def initialize_weights(layers):
    if isinstance(layers, nn.Linear):
         nn.init.normal(layers.weight, 0, 0.01)
         nn.init.constant(layers.bias, 0)
    else:
        for m in layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal(m.weight, 0, 0.01)
                nn.init.constant(m.bias, 0)
                
def build_network():
    pretrained_weights = torch.load(cfg.VGG16_Weights)
    
    pretrained_list = list(pretrained_weights.items())
        
    my_model = AttrNet( num_classes=cfg.num_classes)
    
    my_model_kvpair = my_model.state_dict()
    
    # load ImageNet-trained vgg16_bn weights
    count = 0    
    # load all conv layers (conv1- conv5) and fc1 from pretrained ImageNet weights(79 parameters in total)
    for key, value in my_model_kvpair.items():
        if count < 82: # this is for vgg16 pretrained weights
            my_model_kvpair[key] = pretrained_list[count]
        count+=1

    # initialize fc2,fc3 and second_branch fc
    initialize_weights(my_model.second_branch_fc1)
    initialize_weights(my_model.second_branch_fc2)
    initialize_weights(my_model.fc3_fine_tune)

    return my_model
