import torch
from torchvision import transforms
from loaders.multi_mnist_loader import MNIST
from loaders.cityscapes_loader import CITYSCAPES
from loaders.segmentation_augmentations import *
from loaders.celeba_loader import CELEBA

from loaders.tencent_loader import TENCENT

# Setup Augmentations
cityscapes_augmentations= Compose([RandomRotate(10),
                                   RandomHorizontallyFlip()])

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

def my_collate(batch):
    for i, item in enumerate(batch) :
        if (i==0) :
            data = item[0]
        else :
            data = torch.cat((data, item[0]), 0)
    h = [item[1] for item in batch]
    h = torch.stack(h)
    c = [item[2] for item in batch]
    c = torch.stack(c)
    f = [item[3] for item in batch]
    f = torch.stack(f)
    o = [item[4] for item in batch]
    o = torch.stack(o)
    return [data, h, c, f, o]

def get_dataset(params, configs):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'mnist' in params['dataset']:
        train_dst = MNIST(root=configs['mnist']['path'], train=True, download=True, transform=global_transformer(), multi=True)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)

        val_dst = MNIST(root=configs['mnist']['path'], train=False, download=True, transform=global_transformer(), multi=True)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=100, shuffle=True, num_workers=4)
        return train_loader, train_dst, val_loader, val_dst

    if 'cityscapes' in params['dataset']:
        train_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['train'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']), augmentations=cityscapes_augmentations)
        val_dst = CITYSCAPES(root=configs['cityscapes']['path'], is_transform=True, split=['val'], img_size=(configs['cityscapes']['img_rows'], configs['cityscapes']['img_cols']))

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
        return train_loader, train_dst, val_loader, val_dst

    if 'celeba' in params['dataset']:
        train_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='train', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
        val_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)

        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
        return train_loader, train_dst, val_loader, val_dst
    
    if 'tencent' in params['dataset']:
        train_dst = TENCENT(root=configs['tencent']['path'], is_transform=True, type='train', patch_size=params['patch_size'])
        val_dst = TENCENT(root=configs['tencent']['path'], is_transform=True, type='val', patch_size=params['patch_size'])
        test_dst = TENCENT(root=configs['tencent']['path'], is_transform=True, type='test', patch_size=params['patch_size'])
        
        if (params['patch_size'] > 0) :
            train_loader = torch.utils.data.DataLoader(train_dst, batch_size = params['batch_size'], shuffle=True, num_workers=4, collate_fn=my_collate)
            val_loader = torch.utils.data.DataLoader(val_dst, batch_size = params['batch_size'], num_workers=4, collate_fn=my_collate)
            test_loader = torch.utils.data.DataLoader(val_dst, batch_size = params['batch_size'], num_workers=4, collate_fn=my_collate)
        else :
            train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
            val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
            test_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
        return train_loader, train_dst, val_loader, val_dst, test_loader, test_dst