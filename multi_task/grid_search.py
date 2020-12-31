from main import train_multi_task

def grid_search() :
    ''' 
    hyperparams : 
    - p of dropout [0.25, 0.5, 0.75]
    - lr [0.0001, 0.001, 0.01, 0.1]
    - batch_size [4, 6, 8, 16, 32]
    - img_h [256, 454, 720]
    - patch_size [0, 6, 7, 8]
        if patch_size > 0
            - global_patch = [0, 1]
            - crop_size [(224, 224), (256,256), (256,512)]
        else if patchz_size == 0
            - crop_or_pad [0, 1]
            if crop
            - img_w [454, 806, 1279]
            else if padding
            - img_w [554, 984, 1560]

    avg_out_size ?
    optimiser ?
    decay ?
    decay_epoch ?
    '''
    p_list = [0.25, 0.5, 0.75]
    lr_list = [0.01, 0.001, 0.0001]
    batch_size_list = [32, 16, 8, 6, 4]
    img_h_list = [720, 454, 256]
    img_w_list = []
    patch_size_list = [8, 7, 6, 0]
    global_patch = [True, False]
    crop_size_list = [(256, 512), (256, 256), (224, 224)]
    crop_or_pad = [True, False] # true = pad false = crop

    params = {
        "dataset": "tencent",
        "tasks": ["H", "C", "F", "O"],
        "optimizer": "SGD",
        "normalization_type": "loss+",
        "grid_search": True,
        "train": False,
        "test": False,
        'dropout_prob': 0,
        'lr': 0.001,
        'batch_size': 4,
        'img_h': 225,
        'img_w': 225,
        'patch_size' : 0,
        'global_patch': False,
        'crop_h' : 0,
        'crop_w' : 0,
        'crop_or_pad' : False
    }
    for p in p_list :
        params['dropout_prob'] = p
        for lr in lr_list :
            params['lr'] = lr 
            for bs in batch_size_list :
                params['batch_size'] = bs
                for img_h in img_h_list :
                    params['img_h'] = img_h
                    for ps in patch_size_list :
                        params['patch_size'] = ps
                        if (ps > 0) :
                            for gp in global_patch :
                                params['global_patch'] = gp
                                for cs in crop_size_list :
                                    params['crop_h'] = cs[0]
                                    params['crop_w'] = cs[1]
                        else :
                            for cp in crop_or_pad :
                                params['crop_or_pad'] = cp 
                                if cp == 0 :
                                    for img_w in [454, 806, 1279] :
                                        params['img_w'] = img_w
                                        try:
                                            train_multi_task(params)
                                        except RuntimeError as exception:
                                            if "out of memory" in str(exception):
                                                print("WARNING: out of memory")
                                            if hasattr(torch.cuda, 'empty_cache'):
                                                torch.cuda.empty_cache()
                                            else:
                                                # raise exception
                                                continue
                                else :
                                    for img_w in [554, 984, 1560] :
                                        params['img_w'] = img_w
                                        try:
                                            train_multi_task(params)
                                        except RuntimeError as exception:
                                            if "out of memory" in str(exception):
                                                print("WARNING: out of memory")
                                            if hasattr(torch.cuda, 'empty_cache'):
                                                torch.cuda.empty_cache()
                                            else:
                                                # raise exception
                                                continue



if __name__ == '__main__':
    grid_search()

