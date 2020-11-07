import torch
import glob
import csv
import scipy.misc as m
import numpy as np

from torch.utils import data


class TENCENT(data.Dataset):
    def __init__(self, root, split = "train", is_transform = False, img_size = (32, 32), augmentations = None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.files = {}
        self.labels = {}

        self.all_files = glob.iglob(root + '/original_images/**/*.png', recursive=True)
        #
        # for f in self.all_files:
        #     print(f)

        # print(self.all_files)
        self.files[self.split] = list(map(lambda x: x.split('/')[-2] + '/' + x.split('/')[-1], self.all_files))
        # self.files[self.split] = self.all_files
        # print(self.files[self.split])

        # self.score_color_harmony_file = root + '/subjective_scores_v2/score_color_harmony.csv'
        # self.score_colorfullness_file = root + '/subjective_scores_v2/score_colorfullness.csv'
        # self.score_fitness_file = root + '/subjective_scores_v2/score_fitness.csv'
        # self.score_overall_file = root + '/subjective_scores_v2/score_overall.csv'

        self.all_scores_files = glob.glob(root + '/subjective_scores_v2/*.csv')
        # print(self.all_scores_files)

        label_map = {}
        for file in self.all_scores_files:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader :
                    if row[1] in label_map:
                        label_map[row[1]].append(torch.tensor(float(row[22])))
                    else:
                        label_map[row[1]] = [torch.tensor(float(row[22]))]

        self.labels[self.split] = [label_map[x] for x in label_map]
        # print(self.labels[self.split])
        # print(self.files[self.split])


    def __len__(self):
        # print(self.files[self.split])
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index]
        label_h = self.labels[self.split][index][0]
        label_c = self.labels[self.split][index][0]
        label_f = self.labels[self.split][index][0]
        label_o = self.labels[self.split][index][0]
        # print(self.root + '/original_images/' + img_path)
        img = m.imread(self.root + '/original_images/' + img_path)
        # matplotlib.pyplot.imread
        if self.augmentations is not None:
            img = self.augmentations(np.array(img, dtype=np.uint8))
        
        if self.is_transform:
            img = self.transform_img(img)

        return img, label_h, label_c, label_f, label_o


    def transform_img(self, img):
        img = img.transpose(2, 0, 1)
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
    print(tencent[0])
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