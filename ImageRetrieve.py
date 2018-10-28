import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.autograd import Variable 
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F

import pdb
import os
import argparse
import os
import shutil
import time
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio

from data.data_loader import data_loader
from data.data_processing import DataProcessing

from models.AttrNet import build_network
from utils.count_attr import count_attribute

from models.config import cfg


def main():
        
    arch = cfg.arch

    model = build_network()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("now we are using %d gpus" %torch.cuda.device_count())
    model.cuda()
    print model

    # load the model
    print("=> Loading Network %s" % cfg.resume)
    checkpoint = torch.load(cfg.resume)
    model.load_state_dict(checkpoint['state_dict'])
 
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(cfg.resume, checkpoint['epoch']))
                                 
    cudnn.benchmark = False

    test_loader = data_loader( BatchSize=cfg.batch_size,
                               NumWorkers = cfg.num_workers).test_loader
    print("test data_loader are ready!")
    
    # test mode
    model.eval()

    # test an image
    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((224,224)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # fully connected layers
    FC = []
            
    for iter, test_data in enumerate(test_loader):
        test_inputs, test_labels, u,v = test_data # not use landmarks while testing
        test_inputs, test_labels, u,v =  torch.autograd.Variable(test_inputs.cuda(), volatile=True).float(), torch.autograd.Variable(test_labels.cuda(),volatile=True).float(), torch.autograd.Variable(u.cuda(),volatile=True), torch.autograd.Variable(v.cuda(),volatile=True)

        model_FC = model(test_inputs, u,v)
        if iter % 100 ==0:
            print(model_FC.size())
            print(model_FC)
            sio.savemat('FC.mat', {'FC':FC})
            
        FC.append(model_FC.data.cpu().numpy())

    sio.savemat('FC.mat', {'FC':FC})
    print("Fully Connected Layers are saved as FC.mat ")
        
if __name__ == '__main__':
    main()
    
        
        

        


