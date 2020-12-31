from main import train_multi_task

def _5foldcv() :
    params = {
        "dataset": "tencent",
        "tasks": ["H", "C", "F", "O"],
        "optimizer": "SGD",
        "normalization_type": "loss+",
        "grid_search": False,
        "train": True,
        "test": False,
        'dropout_prob': 0.75,
        'lr': 0.001,
        'batch_size': 6,
        'img_h': 454,
        'img_w': 984,
        'patch_size' : 6,
        'global_patch': True,
        'crop_h' : 272,
        'crop_w' : 590,
        'crop_or_pad' : False
    }

    plcc_list = []
    epoch_list = []
    identifier = ''
    for i in range(0, 5) :
        print(i+1, ' fold')
        identifier, plcc, epoch = train_multi_task(params, i+1)
        plcc_list.append(plcc)
        epoch_list.append(epoch)
    print(plcc_list)
    print(epoch_list)
    best_plcc = max(plcc_list)
    index = plcc_list.index(best_plcc)
    best_epoch = epoch_list[index]
    # test
    params['train'] = False
    params['test'] = True 
    params['exp_identifier'] = identifier
    print('best :', best_epoch, index + 1)   
    # params['best_epoch'] = 26
    # params['best_fold'] = 1
    # params['exp_identifier'] = 'optimizer=SGD|dropout_prob=0.75|lr=0.001|batch_size=8|img_h=454|img_w=984|patch_size=0|global_patch=False|crop_h=256|crop_w=256|crop_or_pad=True'
    params['batch_size'] = 1
    print('test')
    for i in range(0, 5) :
        params['best_epoch'] = epoch_list[i]
        params['best_fold'] = index + 1
        train_multi_task(params)
    
if __name__ == '__main__':
    _5foldcv()