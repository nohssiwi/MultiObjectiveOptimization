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
        'batch_size': 8,
        'img_h': 454,
        'img_w': 984,
        'patch_size' : 0,
        'global_patch': False,
        'crop_h' : 256,
        'crop_w' : 256,
        'crop_or_pad' : True
    }

    plcc_list = []
    epoch_list = []
    id_list = []
    for i in range(0, 5) :
        print(i+1, ' fold')
        identifier, plcc, epoch = train_multi_task(params, i+1)
        plcc_list.append(plcc)
        epoch_list.append(epoch)
        id_list.append(identifier)
    print(plcc_list)
    print(epoch_list)
    best_plcc = max(plcc_list)
    index = plcc_list.index(best_plcc)
    best_epoch = epoch_list[index]
    best_id = id_list[index]
    # test
    params['train'] = False
    params['test'] = True 
    params['best_epoch'] = best_epoch
    params['best_fold'] = index + 1
    params['exp_identifier'] = best_id
    params['batch_size'] = 1
    print('test')
    train_multi_task(params)

if __name__ == '__main__':
    _5foldcv()