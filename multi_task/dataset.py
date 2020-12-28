import torch
from loaders.tencent_loader import TENCENT


def multi_patch_collate(batch):
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
    
    if 'tencent' in params['dataset']:
        train_dst = TENCENT(root=configs['tencent']['path'], is_transform=True, type='train', 
            patch_size=params['patch_size'], img_h=params['img_h'], img_w=params['img_w'], 
            crop_size=(params['crop_h'], params['crop_w']))
        val_dst = TENCENT(root=configs['tencent']['path'], is_transform=True, type='val', 
            patch_size=params['patch_size'], img_h=params['img_h'], img_w=params['img_w'],
            crop_size=(params['crop_h'], params['crop_w']))
        test_dst = TENCENT(root=configs['tencent']['path'], is_transform=True, type='test', 
            patch_size=params['patch_size'], img_h=params['img_h'], img_w=params['img_w'],
            crop_size=(params['crop_h'], params['crop_w']))
        if (params['patch_size'] > 0) :
            train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4, collate_fn=multi_patch_collate)
            val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4, collate_fn=multi_patch_collate)
            test_loader = torch.utils.data.DataLoader(test_dst, batch_size=params['batch_size'], num_workers=4, collate_fn=multi_patch_collate)
        else :
            train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=4)
            val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=4)
            test_loader = torch.utils.data.DataLoader(test_dst, batch_size=params['batch_size'], num_workers=4)
        return train_loader, train_dst, val_loader, val_dst, test_dst, test_loader