from main import train_multi_task

def train_multi_task() :


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
    params = {
        "dataset": "tencent",
        "tasks": ["H", "C", "F", "O"],
        "optimizer": "SGD",
        "normalization_type": "loss+",
        "grid_search": True,
        "train": False,
        "test": False
    }
    for p in [0.25, 0.5, 0.75] :
        params['dropout_prob'] = p
        for lr in [0.1, 0.01, 0.001, 0.0001] :
            params['lr'] = lr 
            for bs in [4, 6, 8, 16, 32] :
                params['batch_size'] = bs
                for img_h in [256, 454, 720] :
                    params['img_h'] = img_h
                    for ps in [0, 6, 7, 8] :
                        params['patch_size'] = ps
                        if (ps > 0) :
                            for gp in [0, 1] :
                                params['global_patch'] = gp
                                for cs in [(224, 224), (256, 256), (256,512)] :
                                    params['crop_h'] = cs[0]
                                    params['crop_w'] = cs[1]
                        else :
                            for cp in [0, 1] :
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
