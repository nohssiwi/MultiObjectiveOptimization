import torch
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import math
from PIL import Image

class TENCENT(data.Dataset):
    def __init__(self, root, type, patch_size, global_patch, crop_or_pad, img_h, img_w, crop_size, fold=0, is_transform=True):
        self.root = root
        self.type = type
        self.fold = fold
        self.is_transform = is_transform
        # self.augmentations = augmentations
        self.patch_size = patch_size
        self.img_h = img_h
        self.img_w = img_w
        self.crop_size = crop_size
        self.global_patch = global_patch
        self.crop_or_pad = crop_or_pad

        data = pd.read_csv(root + '/subjective_scores_v2/all.csv')
        data = data[data['type']!='train']
        cv = pd.read_csv(root + '/subjective_scores_v2/5fold.csv')

        dataset = {
            'cv_train' : [], 
            'cv_val' : [], 
            'val' : [],
            'test' : []
        }

        for index, row in data.iterrows():
            item = {
                'filename' : row['filename'],
                'label_h' : torch.tensor(self.distribution(row['h_0':'h_19'])),
                'label_c' : torch.tensor(self.distribution(row['c_0':'c_19'])),
                'label_f' : torch.tensor(self.distribution(row['f_0':'f_19'])),
                'label_o' : torch.tensor(self.distribution(row['o_0':'o_19']))
            }
            if row['type'] == 'validation' :
                dataset['val'].append(item)
            else :
                dataset['test'].append(item)
                
        for index, row in cv.iterrows():
            item = {
                'filename' : row['filename'],
                'label_h' : torch.tensor(self.distribution(row['h_0':'h_19'])),
                'label_c' : torch.tensor(self.distribution(row['c_0':'c_19'])),
                'label_f' : torch.tensor(self.distribution(row['f_0':'f_19'])),
                'label_o' : torch.tensor(self.distribution(row['o_0':'o_19']))
            }
        
            if row['5fold'] == self.fold :
                dataset['cv_val'].append(item)
            else :
                dataset['cv_train'].append(item)

        self.dataset = dataset


    def distribution(self, row):
        dis = np.zeros(5)
        for v in row :
            v = int(v)
            dis[v-1] = dis[v-1] + 1
        dis = dis / 20
        dis = dis.reshape(-1, 1)
        return dis


    def __len__(self):
        return len(self.dataset[self.type])

    def __getitem__(self, index):
        filename = self.dataset[self.type][index]['filename']
        label_h = self.dataset[self.type][index]['label_h']
        label_c = self.dataset[self.type][index]['label_c']
        label_f = self.dataset[self.type][index]['label_f']
        label_o = self.dataset[self.type][index]['label_o']
        img = Image.open(self.root + '/original_images/' + filename).convert('RGB')
    
        if self.is_transform :
            if (self.patch_size > 0) :
                img = self.transform_img_mp(img)
            else :
                img = self.transform_img(img)

        return img, label_h, label_c, label_f, label_o


    def transform_img(self, img):
        # get width and height of image
        width = img.size[0]
        height = img.size[1]
        # transform
        resize = transforms.Resize(self.img_h)
        toTensor = transforms.ToTensor()
        flip = transforms.RandomHorizontalFlip()
        img = resize(img)
        img = toTensor(img)
        if (width < height) :
            img = img.permute(0, 2, 1)
        if not self.crop_or_pad :
            # cropping
            img = transforms.functional.center_crop(img, (self.img_h, self.img_w))
        else :
            # padding
            w = img.shape[2]
            padding = (math.ceil((self.img_w-w) / 2), 0, math.floor((self.img_w-w) / 2), 0)
            pad = transforms.Pad(padding, fill=0, padding_mode='constant')
            img = pad(img)
        if (self.type == 'train') :
            img = flip(img)
        return img

    def transform_img_mp(self, img):
        # get width and height of image
        width = img.size[0]
        height = img.size[1]
        # transform method
        resize_global = transforms.Resize(self.crop_size)
        resize = transforms.Resize(self.img_h)
        crop = transforms.RandomCrop(self.crop_size)
        toTensor = transforms.ToTensor()
        flip = transforms.RandomHorizontalFlip()
        # transform
        img = toTensor(img)
        # transpose to make sure width > height
        if (width < height) :
            img = img.permute(0, 2, 1)
        # global patch
        if self.global_patch :
            patch_global = resize_global(img)
        img = resize(img)
        # extract patches of image
        patches = []
        for i in range(0, self.patch_size) :  
            patch = crop(img)
            if (self.type == 'train') :
                patch = flip(img)
            patches.append(patch)
        if self.global_patch :
            patches.append(patch_global)
        patches = torch.stack(patches)
        return patches
