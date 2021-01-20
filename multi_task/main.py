import os
import click
import json
import datetime
from timeit import default_timer as timer

import torch
from torch.autograd import Variable
from torch.utils import data

from tqdm import tqdm
from tensorboardX import SummaryWriter

import loss as loss_selector
import dataset as dataset_selector
import metrics as metrics_selector
import model as model_selector
from min_norm_solvers import MinNormSolver, gradient_normalizers

NUM_EPOCHS = 100

# @click.command()
# @click.option('--param_file', default='params.json', help='JSON parameters file')
# def train_multi_task(param_file):
def train_multi_task(params, fold=0):
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    # with open(param_file) as json_params:
    #     params = json.load(json_params)

    exp_identifier = []
    for (key, val) in params.items():
        if ('tasks' in key) or ('dataset' in key) or ('normalization_type' in key) \
            or ('grid_search' in key) or ('train' in key) or ('test' in key):
            continue
        exp_identifier+= ['{}={}'.format(key,val)]

    exp_identifier = '|'.join(exp_identifier)
    # params['exp_id'] = exp_identifier

    if params['train'] :
        train_loader, train_dst, val_loader, val_dst = dataset_selector.get_dataset(params, configs, fold)
        # writer = SummaryWriter(log_dir='5fold_runs/{}_{}'.format(exp_identifier, datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    if params['test'] :
        test_loader, test_dst = dataset_selector.get_dataset(params, configs)

    if params['grid_search'] :
        train_loader, train_dst, val_loader, val_dst = dataset_selector.get_dataset(params, configs, fold)
        # writer = SummaryWriter(log_dir='gs_runs/{}_{}'.format(exp_identifier, datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    loss_fn = loss_selector.get_loss(params)
    metric = metrics_selector.get_metrics(params)

    model = model_selector.get_model(params)
    model_params = []
    model_params_num = 0
    for m in model:
        model_params += model[m].parameters()
        for parameter in model[m].parameters():
            # print('parameter:')
            # print(parameter)
            model_params_num += parameter.numel()
    # print('model params num:')
    # print(model_params_num)

    if 'RMSprop' in params['optimizer']:
        optimizer = torch.optim.RMSprop(model_params, lr=params['lr'])
    elif 'Adam' in params['optimizer']:
        optimizer = torch.optim.Adam(model_params, lr=params['lr'])
    elif 'SGD' in params['optimizer']:
        optimizer = torch.optim.SGD(model_params, lr=params['lr'], momentum=0.9)

    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']
    print('Starting training with parameters \n \t{} \n'.format(str(params)))

    n_iter = 0
    loss_init = {}

    # early stopping
    count = 0
    init_val_plcc = 0
    best_epoch = 0

    # train
    if params['train'] or params['grid_search']:

        for epoch in tqdm(range(NUM_EPOCHS)):
            start = timer()
            print('Epoch {} Started'.format(epoch))
            if (epoch+1) % 30 == 0:
                # Every 30 epoch, half the LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print('Half the learning rate {}'.format(n_iter))

            
            for m in model:
                model[m].train()

            for batch in train_loader:
                n_iter += 1
                
                # First member is always images
                images = batch[0]
                images = Variable(images.cuda())

                labels = {}
                # Read all targets of all tasks
                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels[t] = batch[i+1]
                    labels[t] = Variable(labels[t].cuda())
                    
                # Scaling the loss functions based on the algorithm choice
                loss_data = {}
                grads = {}
                scale = {}
                mask = None
                masks = {}

                # use algo MGDA_UB 
                optimizer.zero_grad()
                # First compute representations (z)
                with torch.no_grad():
                    images_volatile = Variable(images.data)
                rep, mask = model['rep'](images_volatile, mask)
                # As an approximate solution we only need gradients for input
                rep_variable = Variable(rep.data.clone(), requires_grad=True)
                list_rep = False

                # Compute gradients of each loss function wrt z
                for t in tasks:
                    optimizer.zero_grad()
                    out_t, masks[t] = model[t](rep_variable, None)
                    loss = loss_fn[t](out_t, labels[t])
                    loss_data[t] = loss.item()
                    loss.backward()
                    grads[t] = Variable(rep_variable.grad.data.clone(), requires_grad=False)
                    rep_variable.grad.data.zero_()

                # Normalize all gradients, this is optional and not included in the paper.
                gn = gradient_normalizers(grads, loss_data, params['normalization_type'])
                for t in tasks:
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]

                # Frank-Wolfe iteration to compute scales.
                sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                for i, t in enumerate(tasks):
                    scale[t] = float(sol[i])

                # Scaled back-propagation
                optimizer.zero_grad()
                rep, _ = model['rep'](images, mask)
                for i, t in enumerate(tasks):
                    out_t, _ = model[t](rep, masks[t])
                    loss_t = loss_fn[t](out_t, labels[t])
                    loss_data[t] = loss_t.item()
                    if i > 0:
                        loss = loss + scale[t]*loss_t
                    else:
                        loss = scale[t]*loss_t
                loss.backward()
                optimizer.step()

                # writer.add_scalar('training_loss', loss.item(), n_iter)
                # for t in tasks:
                #     writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)
            
            # validation
            for m in model:
                model[m].eval()

            tot_loss = {}
            tot_loss['all'] = 0.0
            met = {}
            for t in tasks:
                tot_loss[t] = 0.0
                met[t] = 0.0

            num_val_batches = 0
            for batch_val in val_loader:
                with torch.no_grad():
                    val_images = Variable(batch_val[0].cuda())
                labels_val = {}

                for i, t in enumerate(all_tasks):
                    if t not in tasks:
                        continue
                    labels_val[t] = batch_val[i+1]
                    with torch.no_grad():
                        labels_val[t] = Variable(labels_val[t].cuda())

                val_rep, _ = model['rep'](val_images, None)
                for t in tasks:
                    out_t_val, _ = model[t](val_rep, None)
                    loss_t = loss_fn[t](out_t_val, labels_val[t])
                    tot_loss['all'] += loss_t.item()
                    tot_loss[t] += loss_t.item()
                    metric[t].update(out_t_val, labels_val[t])
                num_val_batches+=1

            avg_plcc = 0
            for t in tasks:
                # writer.add_scalar('validation_loss_{}'.format(t), tot_loss[t]/num_val_batches, n_iter)
                metric_results = metric[t].get_result()
                avg_plcc += metric_results['plcc']
                metric_str = 'task_{} : '.format(t)
                for metric_key in metric_results:
                    # writer.add_scalar('metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter)
                    metric_str += '{} = {}  '.format(metric_key, metric_results[metric_key])
                metric[t].reset()
                metric_str += 'loss = {}'.format(tot_loss[t]/num_val_batches)
                print(metric_str)
            print('all loss = {}'.format(tot_loss['all']/len(val_dst)))
            # writer.add_scalar('validation_loss', tot_loss['all']/len(val_dst), n_iter)
            avg_plcc /= 4

            print(avg_plcc)
            print(init_val_plcc)
            if init_val_plcc < avg_plcc:
                init_val_plcc = avg_plcc
                # save model weights if val loss decreases
                print('Saving model...')
                state = {'epoch': epoch+1,
                        'model_rep': model['rep'].state_dict(),
                        'optimizer_state' : optimizer.state_dict()}
                for t in tasks:
                    key_name = 'model_{}'.format(t)
                    state[key_name] = model[t].state_dict()

                torch.save(state, "saved_models/{}_{}_{}_model.pkl".format(exp_identifier, epoch+1, fold))
                best_epoch = epoch + 1
                # reset count
                count = 0
            elif init_val_plcc >= avg_plcc:
                count += 1
                if count == 10:
                    print('Val EMD loss has not decreased in %d epochs. Training terminated.' % 10)
                    break

            end = timer()
            print('Epoch ended in {}s'.format(end - start))

        print('Training completed.')
    # return exp_identifier, init_val_plcc, best_epoch
    return init_val_plcc


    # # test
    # if params['test'] :
    #     state = torch.load(os.path.join('./saved_models', "{}_{}_{}_model.pkl".format(params['exp_identifier'], params['best_epoch'], params['best_fold'])))
    #     model['rep'].load_state_dict(state['model_rep'])
    #     for t in tasks :
    #         key_name = 'model_{}'.format(t)
    #         model[t].load_state_dict(state[key_name])
    #     print('Successfully loaded {}_{}_{}_model'.format(params['exp_identifier'], params['best_epoch'], params['best_fold']))


    #     for m in model:
    #         model[m].eval()
        
    #     test_tot_loss = {}
    #     test_tot_loss['all'] = 0.0
    #     test_met = {}
    #     for t in tasks:
    #         test_tot_loss[t] = 0.0
    #         test_met[t] = 0.0

    #     num_test_batches = 0
    #     for batch_test in test_loader:
    #         with torch.no_grad():
    #             test_images = Variable(batch_test[0].cuda())
    #         labels_test = {}

    #         for i, t in enumerate(all_tasks):
    #             if t not in tasks:
    #                 continue
    #             labels_test[t] = batch_test[i+1]
    #             with torch.no_grad():
    #                 labels_test[t] = Variable(labels_test[t].cuda())

    #         test_rep, _ = model['rep'](test_images, None)
    #         for t in tasks:
    #             out_t_test, _ = model[t](test_rep, None)
    #             test_loss_t = loss_fn[t](out_t_test, labels_test[t])
    #             test_tot_loss['all'] += test_loss_t.item()
    #             test_tot_loss[t] += test_loss_t.item()
    #             metric[t].update(out_t_test, labels_test[t])
    #         num_test_batches+=1

    #     print('test:')
    #     for t in tasks:
    #         test_metric_results = metric[t].get_result()
    #         test_metric_str = 'task_{} : '.format(t)
    #         for metric_key in test_metric_results:
    #             test_metric_str += '{} = {}  '.format(metric_key, test_metric_results[metric_key])
    #         metric[t].reset()
    #         # test_metric_str += 'loss = {}'.format(test_tot_loss[t]/num_test_batches)
    #         print(test_metric_str)
    #     # print('all loss = {}'.format(test_tot_loss['all']/len(test_dst)))
        
        


if __name__ == '__main__':
    train_multi_task()
