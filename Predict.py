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

from data.data_loader import data_loader
from data.data_processing import DataProcessing

from models.AttrNet import build_network
from utils.count_attr import count_attribute

from models.config import cfg


#attr = []
# read attribute index and attribute name
#with open(cfg.ATTR_FILE) as f:
#    for line in f:
#        parts = line.split()
#        attr_name = parts[0]
#        attr.append(attr_name)
#f.close()

top3_P = [0]*(cfg.num_classes)
top5_P =[0]*(cfg.num_classes)
top10_P = [0]*(cfg.num_classes)
top3_TP = [0]*(cfg.num_classes)
top5_TP = [0]*(cfg.num_classes)
top10_TP = [0]*(cfg.num_classes)
top3_precision = [0.0]*(cfg.num_classes)
top5_precision =  [0.0]*(cfg.num_classes)
top10_precision = [0.0]*(cfg.num_classes)     
top3_file = open('top3_precision.txt','w')
top5_file = open('top5_precision.txt', 'w')
top10_file = open('top10_precision.txt', 'w')

# calculate precision
def cal_precision(outputs, targets):
    
    batch_size = cfg.batch_size
    label_size = len(targets[0])
    
    for i in range(len(targets)): # the ith img
        target = targets[i].data
        output = outputs[i].data
        predicted = (F.sigmoid(output.cpu())).numpy()

        index = np.argsort(predicted)
       
        top3_index = []
        top5_index = []
        top10_index = []
        for i in range(1,11):
            topi = index[label_size-i]
            if 1<=i<=3:
                top3_index.append(topi)
                top3_P[topi] += 1
                
            if 1<=i<=5:
                top5_index.append(topi)
                top5_P[topi] +=1
                
            if 1<=i<=10:
                top10_index.append(topi)
                top10_P[topi]+=1

        # rule out the images that don't have any attribute
        if sum(target) == 0:
            continue # move to the next image

        for attr_idx,attr in enumerate(target):
            if attr == 1: # true
                if attr_idx in top3_index:
                    top3_TP[attr_idx]+=1    
                if attr_idx in top5_index:
                    top5_TP[attr_idx]+=1
                if attr_idx in top10_index:
                    top10_TP[attr_idx] += 1

def print_precision(batch):
    top3_batch =  [0.0]*(cfg.num_classes)
    top5_batch = [0.0]*(cfg.num_classes)
    top10_batch = [0.0]*(cfg.num_classes)
    for idx, attr in enumerate(top3_P):
        top3_batch[idx] = float(top3_TP[idx])/top3_P[idx] if top3_P[idx]!=0 else 0
        top5_batch[idx] = float(top5_TP[idx])/top5_P[idx] if top5_P[idx]!=0 else 0
        top10_batch[idx] = float(top10_TP[idx])/top10_P[idx] if top10_P[idx]!=0 else 0

    print("=============== t o p 3 precision =====================")
    for idx, p in enumerate(top3_batch):
        print("attr[%d]: %.4f"%(idx, top3_batch[idx]))
    print('\n')

    print("=============== t o p 5 precision =====================")
    for idx, p in enumerate(top5_batch):
        print("attr[%d]: %.4f"%(idx, top5_batch[idx]))
    print('\n')

    print("=============== t o p 10 precision =====================")
    for idx, p in enumerate(top10_batch):
        print("attr[%d]: %.4f"%(idx, top10_batch[idx]))
    print('\n')
    

def main():
        
    arch = cfg.arch

    model = build_network()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print("now we are using %d gpus" %torch.cuda.device_count())
    model.cuda()
    print model

    model_state_dict = model.module.state_dict() if torch.cuda.device_count()>1 else model.state_dict()
    print('model',len(model_state_dict))

    # load the model
    print("=> Loading Network %s" % cfg.resume)
    checkpoint = torch.load(cfg.resume)
    print('checkpoint', len(checkpoint['state_dict']))
          
    model.load_state_dict(checkpoint['state_dict'])
    #pretrained_list = list(checkpoint['state_dict'].items())
    #my_model_pair = model.state_dict()

    
    #count = 0
    #for k,v in my_model_pair.items():
    #    my_model_pair[k] = pretrained_list[count]
    #    count += 1
    #model.load_state_dict(checkpoint['state_dict'])

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

            
    for iter, test_data in enumerate(test_loader):
        test_inputs, test_labels, u,v = test_data # not use landmarks while testing
        test_inputs, test_labels, u,v =  torch.autograd.Variable(test_inputs.cuda(), volatile=True).float(), torch.autograd.Variable(test_labels.cuda(),volatile=True).float(), torch.autograd.Variable(u.cuda(),volatile=True), torch.autograd.Variable(v.cuda(),volatile=True)

        test_outputs = model(test_inputs, u,v)
        cal_precision(test_outputs, test_labels)
        print_precision(iter)

    zero_top3_idx= []
    zero_top5_idx= []
    zero_top10_idx = []
    for idx, attr in enumerate(top3_P):
        if top3_P[idx]==0:
            zero_top3_idx.append(idx)
        if top5_P[idx]==0:
            zero_top5_idx.append(idx)
        if top10_P[idx]==0:
            zero_top10_idx.append(idx)
        top3_precision[idx] = float(top3_TP[idx])/top3_P[idx] if top3_P[idx]!=0 else 0
        top5_precision[idx] = float(top5_TP[idx])/top5_P[idx] if top5_P[idx]!=0 else 0
        top10_precision[idx] = float(top10_TP[idx])/top10_P[idx] if top10_P[idx]!=0 else 0
    
    print("=============== t o p 3 precision =====================")
    for idx, p in enumerate(top3_precision):
        print("attr[%d]: %.4f"%(idx, top3_precision[idx]))
        top3_file.write("attr[%d]: %.4f\n"%(idx, top3_precision[idx]))
    print('\n')

    
    print("=============== t o p 5 precision =====================")
    for idx, p in enumerate(top5_precision):
        print("attr[%d]: %.4f"%(idx, top5_precision[idx]))
        top5_file.write("attr[%d]: %.4f\n"%(idx, top5_precision[idx]))
    print('\n')

    
    print("=============== t o p 10 precision =====================")
    for idx, p in enumerate(top10_precision):
        print("attr[%d]: %.4f"%(idx, top10_precision[idx]))
        top10_file.write("attr[%d]: %.4f\n"%(idx, top10_precision[idx]))
    print('\n')

    avg_top3_precision = float(sum(top3_precision))/(len(top3_precision)-len(zero_top3_idx))
    avg_top5_precision = float(sum(top5_precision))/(len(top5_precision)-len(zero_top5_idx))
    avg_top10_precision = float(sum(top10_precision))/(len(top10_precision)-len(zero_top10_idx))
    print('avg_top3_precision: %.4f'%avg_top3_precision)
    print('avg_top5_precision: %.4f'% avg_top5_precision)
    print('avg_top10_precision: %.4f'% avg_top10_precision)
    top3_file.write('avg_top3_precision: %.4f'%avg_top3_precision)
    top5_file.write('avg_top5_precision: %.4f'%avg_top5_precision)
    top10_file.write('avg_top10_precision: %.4f'%avg_top10_precision)

    # True Positive
    print('------------------Top3 True Positive------------------')
    for i in range(len(top3_TP)):
        print("%d"% top3_TP[i])
        print(' ')

    print('------------------Top5 True Positive------------------')
    for i in range(len(top5_TP)):
        print("%d"% top5_TP[i])
        print(' ')

    print('------------------Top10 True Positive------------------')
    for i in range(len(top10_TP)):
        print("%d"% top10_TP[i])
        print(' ')
        
    # recall rate
    ground_truth = np.loadtxt('utils/ground_truth.txt', dtype=np.int)
    top3_recall = np.zeros(cfg.num_classes)
    top5_recall = np.zeros(cfg.num_classes)
    top10_recall = np.zeros(cfg.num_classes)

    top3_file.write('--------------------- recall -----------------------')
    top3_file.write('\n')
    top5_file.write('--------------------- recall -----------------------')
    top5_file.write('\n')
    top10_file.write('--------------------- recall -----------------------')
    top10_file.write('\n')
    print('----------------- recall rate ---------------------')
    for i,t in enumerate(top3_TP):
        top3_recall[i] = float(top3_TP[i])/ground_truth[i]
        print("%.4f"%top3_recall[i])
        top3_file.write("%.4f"%top3_recall[i])
        top3_file.write(' ')

    print("avg top3 recall %.4f" % (sum(top3_recall)/float(len(top3_recall))))
    
    for i,t in enumerate(top5_TP):
        top5_recall[i] = float(top5_TP[i])/ground_truth[i]
        print("%.4f"%top5_recall[i])
        top5_file.write("%.4f"%top5_recall[i])
        top5_file.write(' ')
    print("avg top5 recall %.4f" % (sum(top5_recall)/float(len(top5_recall))))
    
    for i,t in enumerate(top10_TP):
        top10_recall[i] = float(top10_TP[i])/ground_truth[i]
        print("%.4f"%top10_recall[i])
        top10_file.write("%.4f"%top10_recall[i])
        top10_file.write(' ')
    print("avg top10 recall %.4f" % (sum(top10_recall)/float(len(top10_recall))))
                                
    top3_file.close()
    top5_file.close()
    top10_file.close()
    
if __name__ == '__main__':
    main()
    
        
        

        


