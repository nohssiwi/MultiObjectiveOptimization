import torch
import glob
import csv
import scipy.misc as m
import numpy as np
import torchvision.transforms as transforms

from PIL import Image


from torch.utils import data


class TENCENT(data.Dataset):
    def __init__(self, root, type, patch_size = 10, is_transform = True):
        self.root = root
        self.type = type
        self.is_transform = is_transform
        # self.augmentations = augmentations
        self.patch_size = patch_size
        self.files = {
            'train' : [],
            'val' : []
        }
        self.labels = {
            'train' : [],
            'val' : []
        }

        dataset = {}
        all_scores_files = glob.glob(root + '/subjective_scores_v2/*.csv')
        for file in all_scores_files:
            # print(file)
            dimension = file.split('/')[-1].split('_')[1].split('.')[0]
            # print(dimension)
            key = ''
            if dimension == 'overall' :
                key = 'o'
            elif dimension == 'color' :
                key = 'h'
            elif dimension == 'colorfulness' :
                key = 'c'
            elif dimension == 'fineness' :
                key = 'f'
            with open(file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader :
                    dis = self.distribution(row[2:22])
                    dis = dis.reshape(-1, 1)
                    index = row[1].find('img')
                    filename = row[1][:index-1] + '/' + row[1][index:]
                    if row[1] in dataset:
                        dataset[row[1]]['labels'][key] = torch.tensor(dis)
                    else:
                        dataset[row[1]] = {
                            'filename' : filename,
                            'labels' : {key : torch.tensor(dis)}
                        }
        dataset = [dataset[x] for x in dataset]
        # print(dataset)

        # divide dataset
        # total: 1091 train: 872 val: 219
        for i in range(0, 872):
            self.files['train'].append(dataset[i]['filename'])
            self.labels['train'].append(dataset[i]['labels'])
        for i in range(872, 1091):
            self.files['val'].append(dataset[i]['filename'])
            self.labels['val'].append(dataset[i]['labels'])

        # print(self.files['train'])
        # print(self.files['val'])
        # print(len(self.labels['train']))
        # print(self.labels['val'])

    def distribution(self, row):
        dis = np.zeros(5)
        for v in row :
            v = int(v)
            dis[v-1] = dis[v-1] + 1
        dis = dis / 20
        return dis


    def __len__(self):
        return len(self.files[self.type])

    def __getitem__(self, index):
        img_path = self.files[self.type][index]
        label_h = self.labels[self.type][index]['h']
        label_c = self.labels[self.type][index]['c']
        label_f = self.labels[self.type][index]['f']
        label_o = self.labels[self.type][index]['o']
        # print(self.root + '/original_images/' + img_path)
        # img = m.imread(self.root + '/original_images/' + img_path)
        img = Image.open(self.root + '/original_images/' + img_path).convert('RGB')
        if self.is_transform :
            img = self.transform_img(img)
        
        
        # matplotlib.pyplot.imread
        # if self.augmentations is not None:
        #     img = self.augmentations(np.array(img, dtype=np.uint8))
        
        # if self.is_transform:
            

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
        crop = transforms.RandomCrop(256)
        img = resize(img)
        # extract patches of image
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
        # to tensor
        # image = torch.from_numpy(patches).float()
        
        return patches


if __name__ == '__main__':
    
    local_path = '../../Qomex_2020_mobile_game_imges'
    tencent = TENCENT(local_path, type = 'train')
    print(tencent[0])
    trainloader = data.DataLoader(tencent, batch_size=4, num_workers=0)
    # print(trainloader)