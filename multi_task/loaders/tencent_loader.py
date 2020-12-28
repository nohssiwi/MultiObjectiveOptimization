import torch
import glob
import csv
import scipy.misc as m
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
import math
from PIL import Image


from torch.utils import data


class TENCENT(data.Dataset):
    def __init__(self, root, type, patch_size, img_h, img_w, is_transform = True):
        self.root = root
        self.type = type
        self.is_transform = is_transform
        # self.augmentations = augmentations
        self.patch_size = patch_size
        self.img_h = img_h
        self.img_w = img_w

        data = pd.read_csv(root + '/subjective_scores_v2/all.csv')

        dataset = {
            'train' : [], 
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
            if row['type'] == 'train' :
                dataset['train'].append(item)
            elif row['type'] == 'validation' :
                dataset['val'].append(item)
            else :
                dataset['test'].append(item)

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
            img = self.transform_img(img)

        return img, label_h, label_c, label_f, label_o


    def transform_img(self, img):
        # get width and height of image
        width = img.size[0]
        height = img.size[1]
        resize = transforms.Resize(self.img_h)
        img = resize(img)
        
        toTensor = transforms.ToTensor()
        img = toTensor(img)

        if (height > width) :
            img = img.permute(0, 2, 1)
        w = img.shape[2]
        padding = (math.ceil((img_w-w) / 2), 0, math.floor((img_w-w) / 2), 0)
        pad = transforms.Pad(padding, fill=0, padding_mode='constant')
        img = pad(img)
        
        image = img
        return image


if __name__ == '__main__':
    
    local_path = '../../Qomex_2020_mobile_game_imges'
    tencent = TENCENT(local_path, type = 'train')
    print(tencent[0])
    trainloader = data.DataLoader(tencent, batch_size=4, num_workers=0)
    # print(trainloader)