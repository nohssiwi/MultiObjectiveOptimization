import torch
import glob
import csv
import scipy.misc as m
import numpy as np
import torchvision.transforms as transforms

from torch.utils import data


class TENCENT(data.Dataset):
    def __init__(self, root, split = "train", is_transform = False, img_size = (32, 32), augmentations = None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
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
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index]
        label_h = self.labels[self.split][index]['h']
        label_c = self.labels[self.split][index]['c']
        label_f = self.labels[self.split][index]['f']
        label_o = self.labels[self.split][index]['o']
        # print(self.root + '/original_images/' + img_path)
        img = m.imread(self.root + '/original_images/' + img_path)
        # matplotlib.pyplot.imread
        if self.augmentations is not None:
            img = self.augmentations(np.array(img, dtype=np.uint8))
        
        if self.is_transform:
            img = self.transform_img(img)

        return img, label_h, label_c, label_f, label_o


    def transform_img(self, img):
        img = img[:,:,:3]
        # get height
        h = img.shape[0]
        # get width
        w = img.shape[1]
        # if h > w :
        #     img = img.transpose(2, 1, 0)
        # else :
        #     img = img.transpose(2, 0, 1)
        # max_h = 1125
        # max_w = 2436

        # if img.shape[1] < max_h :
        #     padding = (max_h - img.shape[1]) / 2
        #     p1 = int(padding)
        #     p2 = max_h - p1 - img.shape[1]
        #     img = np.pad(img, ((0, 0), (p1, p2), (0, 0)), 'constant', constant_values = (0,0))
        #
        # if img.shape[2] < max_w :
        #     padding = (max_w - img.shape[2]) / 2
        #     p1 = int(padding)
        #     p2 = max_w - p1 - img.shape[2]
        #     img = np.pad(img, ((0, 0), (0, 0), (p1, p2)), 'constant', constant_values = (0,0))
        # img = img.transpose(1, 2, 0)
        # img -= self.mean
        img = m.imresize(img, (int(h / 5), int(w / 5)))
        img = img.transpose(2, 0, 1)
        # Resize scales images from 0 to 255, thus we need to divide by 255.0
        # img = img.astype(float) / 255.0
        img = torch.from_numpy(img).float()
        return img




if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt
    '''
    2,160 x 1,080‬
    ‪2,340 x 1,080‬
    '''
    bs = 4
    
    local_path = '../../Qomex_2020_mobile_game_imges'
    tencent = TENCENT(local_path, is_transform=True, augmentations=None)
    # print(tencent[0])
    trainloader = data.DataLoader(tencent, batch_size=4, num_workers=0)
    # print(trainloader)

    # for i, data in enumerate(trainloader):
    #     imgs = data[0]
    #     labels = data[1:]
    #     imgs = imgs.numpy()[:, ::-1, :, :]
        
    #     # imgs = np.transpose(imgs, [0,2,3,1])

    #     f, axarr = plt.subplots(bs,4)
    #     for j in range(bs):
    #         axarr[j][0].imshow(imgs[j])
    #         axarr[j][1].imshow(imgs[j])
    #         axarr[j][2].imshow(imgs[j])
    #         axarr[j][3].imshow(imgs[j])
    #         # axarr[j][1].imshow(tencent.decode_segmap(labels.numpy()[j]))
    #         # axarr[j][2].imshow(instances[j,0,:,:])
    #         # axarr[j][3].imshow(instances[j,1,:,:])
    #     plt.show()
    #     a = input()
    #     if a == 'ex':
    #         break
    #     else:
    #         plt.close()