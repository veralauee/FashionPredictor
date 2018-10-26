"""
Created on Wed Aug 01 2018
@author: Xin Liu, Ziwei Liu

Create a DataProcessing class here, which is aimed to crop the region based on its bbox, and then to resize images
"""


import torch
import os
import sys
from skimage import io
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
import models.config as cfg
import cv2
from pdb import set_trace as bp

    
class DataProcessing(Dataset):
    def __init__(self, data_path, img_path, img_filename,label_filename, transform, bbox_file, landmarks_file, iuv_file, img_size):

        self.data_path = data_path
        self.img_path = os.path.join(data_path,img_path)
        self.transform = transform 

        # read img file from file ('train.txt', 'val.txt','test.txt')
        img_filepath = img_filename
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()

        # read iuv img file from file
        iuv_filepath = iuv_file
        fp_iuv = open(iuv_filepath, 'r')
        self.iuv_filename = [x.strip() for x in fp_iuv]
        fp_iuv.close()

        #read labels from file
        label_filepath = label_filename
        labels = np.loadtxt(label_filepath, dtype=np.int64)
        
        self.label = labels

        # load the whole bboxes as matrix
        self.bbox_file = bbox_file
        if os.path.exists(self.bbox_file):
            self.bboxes = np.loadtxt(bbox_file, usecols=(0,1,2,3))
        else:
            self.bboxes = None
            
        # load the rois
        if os.path.exists(landmarks_file):
            self.landmarks = np.loadtxt(landmarks_file)
        else:
            self.landmarks = None
            
        self.img_size = img_size
        

    # read img, select bounding_box region, and read its label
    def __getitem__(self,idx):
        
        # crop the bbox region
        if len(self.bboxes)>0:
            bbox_cor = self.bboxes[idx]
            x1 = max(0,int(bbox_cor[0])-10)
            y1 = max(0,int(bbox_cor[1])-10)
            x2 = int(bbox_cor[2])+10
            y2 = int(bbox_cor[3])+10
        
        img = Image.open(os.path.join(self.data_path,self.img_filename[idx]))
        imgpath = os.path.join(self.data_path,self.img_filename[idx])
        
        if len(self.bboxes)>0:
            img = img.crop(box=(x1,y1,x2,y2))
                        
        # resize img
        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = img.convert('RGB') 
       
        if self.transform is not None:
            img = self.transform(img)
        
        label = torch.from_numpy(self.label[idx])
        
        #if self.landmarks:
        #    landmarks = torch.from_numpy(self.landmarks[idx])
        #else:
        #    landmarks = None
            
        # read iuv image, get iuv corordinates
        iuv_img = cv2.imread(os.path.join(self.data_path, self.iuv_filename[idx]))        
        if iuv_img is None: # densepose cannot generate iuv, use 0 instead
            iuv_img = np.zeros((224,224,3))
            
        if len(self.bboxes)>0:
            iuv = iuv_img[y1:y2, x1:x2]
        else:
            iuv = iuv_img

        orig_u = iuv[:,:,1]
        orig_v = iuv[:,:,2]

        iuv_w, iuv_h,channels = iuv_img.shape
        list_u = []
        list_v = []
        for partind in xrange(1,23): # no face part
            x,y=np.where(iuv[:,:,0]==partind)
            
            u_points = orig_u[x,y] # a list
            v_points = orig_v[x,y]

            if len(u_points)<=0 or len(v_points)<=0:
                list_u.append(-1)
                list_v.append(-1)
                continue
            u_cnt = [0]*14
            v_cnt = [0]*14
            
            for u_i in range(len(u_points)):
                u_cor = int(14*float(u_points[u_i])/iuv_w)            
                if u_cor >13:
                    u_cor = 13
                u_cnt[u_cor] += 1

            for v_i in range(len(v_points)):
                v_cor = int(14*float(v_points[v_i])/iuv_h)
                if v_cor >13:
                    v_cor = 13
                v_cnt[v_cor] +=1
                                
            list_u.append(u_cnt.index(max(u_cnt)))
            list_v.append(v_cnt.index(max(v_cnt)))
        #print(list_u)
        #print(list_u.shape)
        #print(list_v)
        #print(list_v.shape)
        u = torch.from_numpy(np.asarray(list_u))
        v = torch.from_numpy(np.asarray(list_v))
        return img, label, u,v

        
    def __len__(self):
        return len(self.img_filename)
