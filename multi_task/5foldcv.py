from main import train_multi_task
import torch

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
        'patch_size' : 4,
        'global_patch': True,
        'crop_h' : 340,
        'crop_w' : 738,
        'crop_or_pad' : True
    }

    plcc_list = []
    epoch_list = []
    identifier = ''
    plcc = 0
    epoch = 0
    # for i in range(0, 5) :
    i = 2
    # print(i+1, ' fold')
    # try:
    #     identifier, plcc, epoch = train_multi_task(params, i+1)
    # except RuntimeError as exception:
    #     if "out of memory" in str(exception):
    #         print("WARNING: out of memory")
    #     if hasattr(torch.cuda, 'empty_cache'):
    #         torch.cuda.empty_cache()
    #     else:
    #         raise exception
    # # identifier, plcc, epoch = train_multi_task(params, i+1)
    # plcc_list.append(plcc)
    # epoch_list.append(epoch)

    # print(plcc_list)
    # print(epoch_list)
    # best_plcc = max(plcc_list)
    # index = plcc_list.index(best_plcc)
    # best_epoch = epoch_list[index]
    # test
    params['train'] = False
    params['test'] = True 
    # params['exp_identifier'] = identifier
    # print('best :', best_epoch, index + 1)   
    params['exp_identifier'] = 'optimizer=SGD|dropout_prob=0.75|lr=0.001|batch_size=6|img_h=454|img_w=984|patch_size=4|global_patch=True|crop_h=340|crop_w=738|crop_or_pad=True'
    params['batch_size'] = 1
    print('test')
    # for i in range(0, 5) :
    # params['best_epoch'] = epoch_list[i]
    params['best_epoch'] = 33
    # params['best_fold'] = i + 1
    params['best_fold'] = 3
    train_multi_task(params)
    
if __name__ == '__main__':
    _5foldcv()