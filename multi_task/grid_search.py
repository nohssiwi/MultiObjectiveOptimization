from main import train_multi_task
import torch
import optuna

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
                                    for img_w in [1279, 806, 454] :
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
                                    for img_w in [1560, 984, 554] :
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


def objective(trial):

    params = {
        "dataset": "tencent",
        "tasks": ["H", "C", "F", "O"],
        "optimizer": "SGD",
        "normalization_type": "loss+",
        "grid_search": True,
        "train": False,
        "test": False,
        'dropout_prob': trial.suggest_float('dropout_prob', 0.1, 0.9),
        'lr': trial.suggest_float('lr', 1e-5, 1e-1),
        # 'dropout_prob': 0.75,
        # 'lr': 0.001,
        'batch_size': 4,
        'img_h': 454,
        'img_w': 984,
        'patch_size' : 0,
        'global_patch': True,
        'crop_h' : 340,
        'crop_w' : 738,
        'crop_or_pad' : True
    }

    # _, plcc, _ = train_multi_task(params)
    plcc = train_multi_task(params, 1)
    return plcc


if __name__ == '__main__':
    # grid_search()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)



  # if params['grid_search'] :
        #     # val_loader, val_dst = dataset_selector.get_dataset(params, configs)
        #     for m in model:
        #         model[m].train()

        #     for batch in val_loader:
        #         n_iter += 1
        #         # First member is always images
        #         images = batch[0]
        #         images = Variable(images.cuda())
        #         labels = {}
        #         # Read all targets of all tasks
        #         for i, t in enumerate(all_tasks):
        #             if t not in tasks:
        #                 continue
        #             labels[t] = batch[i+1]
        #             labels[t] = Variable(labels[t].cuda())
        #         # Scaling the loss functions based on the algorithm choice
        #         loss_data = {}
        #         grads = {}
        #         scale = {}
        #         mask = None
        #         masks = {}
        #         # use algo MGDA_UB 
        #         optimizer.zero_grad()
        #         # First compute representations (z)
        #         with torch.no_grad():
        #             images_volatile = Variable(images.data)
        #         rep, mask = model['rep'](images_volatile, mask)
        #         # As an approximate solution we only need gradients for input
        #         rep_variable = Variable(rep.data.clone(), requires_grad=True)
        #         list_rep = False
        #         # Compute gradients of each loss function wrt z
        #         for t in tasks:
        #             optimizer.zero_grad()
        #             out_t, masks[t] = model[t](rep_variable, None)
        #             loss = loss_fn[t](out_t, labels[t])
        #             loss_data[t] = loss.item()
        #             loss.backward()
        #             grads[t] = Variable(rep_variable.grad.data.clone(), requires_grad=False)
        #             rep_variable.grad.data.zero_()
        #         # Normalize all gradients, this is optional and not included in the paper.
        #         gn = gradient_normalizers(grads, loss_data, params['normalization_type'])
        #         for t in tasks:
        #             for gr_i in range(len(grads[t])):
        #                 grads[t][gr_i] = grads[t][gr_i] / gn[t]
        #         # Frank-Wolfe iteration to compute scales.
        #         sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
        #         for i, t in enumerate(tasks):
        #             scale[t] = float(sol[i])
        #         # Scaled back-propagation
        #         optimizer.zero_grad()
        #         rep, _ = model['rep'](images, mask)
        #         for i, t in enumerate(tasks):
        #             out_t, _ = model[t](rep, masks[t])
        #             loss_t = loss_fn[t](out_t, labels[t])
        #             loss_data[t] = loss_t.item()
        #             metric[t].update(out_t, labels[t])
        #             if i > 0:
        #                 loss = loss + scale[t]*loss_t
        #             else:
        #                 loss = scale[t]*loss_t
        #         loss.backward()
        #         optimizer.step()

        #     avg_plcc_list = []
        #     avg_plcc = 0
        #     writer.add_scalar('training_loss', loss.item(), n_iter)
        #     for t in tasks:
        #         writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)
        #         metric_results = metric[t].get_result()
        #         avg_plcc += metric_results['plcc']
        #         metric_str = 'task_{} : '.format(t)
        #         for metric_key in metric_results:
        #             writer.add_scalar('training_metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter)
        #             metric_str += '{} = {}  '.format(metric_key, metric_results[metric_key])
        #         metric[t].reset()
        #         print(metric_str)
        #     avg_plcc /= 4
        #     avg_plcc_list.append(avg_plcc) 