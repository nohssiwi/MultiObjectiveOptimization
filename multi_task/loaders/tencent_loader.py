import torch
import glob
import csv
import scipy.misc as m
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image


from torch.utils import data


class TENCENT(data.Dataset):
    def __init__(self, root, type, patch_size, is_transform = True):
        self.root = root
        self.type = type
        self.is_transform = is_transform
        # self.augmentations = augmentations
        self.patch_size = patch_size

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
        # if (width > height) :
        #     crop = transforms.RandomCrop((200, 400))
        # else :
        #     crop = transforms.RandomCrop((400, 200))
        resize = transforms.Resize(454)
        img = resize(img)
        w = img.size[0]
        h = img.size[1]
        if (w > h) :
            # (454, 984)
            padding = (int((984-w) / 2), 0)
        else :
            padding = (0, int((984-h) / 2))
        pad = transforms.Pad(padding, fill=0, padding_mode='constant')
        crop = transforms.RandomCrop(128)
        
        # extract patches of image
        if (self.patch_size > 0) :
            patches = []
            for i in range(0, self.patch_size) :       
                patch = crop(img)
                patch = np.array(patch, dtype = np.uint8)
                patches.append(patch)
            patches = np.array(patches)

            # transpose to make sure width > height
            # if (width > height) :
            #     patches = patches.transpose(0, 3, 1, 2)
            # else :
            #     patches = patches.transpose(0, 3, 2, 1)
            patches = patches.transpose(0, 3, 1, 2)
            image = patches
        else :
            img = pad(img)
            toTensor = transforms.ToTensor()
            img = toTensor(img)
            # img = np.array(img)
            if (w > h) :
                img = img.transpose(2, 0, 1)
            else :
                img = img.transpose(2, 1, 0)
            # image = torch.from_numpy(img).float()
            image = img

        return image


if __name__ == '__main__':
    
    local_path = '../../Qomex_2020_mobile_game_imges'
    tencent = TENCENT(local_path, type = 'train')
    print(tencent[0])
    trainloader = data.DataLoader(tencent, batch_size=4, num_workers=0)
    # print(trainloader)