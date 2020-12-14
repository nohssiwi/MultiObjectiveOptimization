# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

from losses import l1_loss_instance
import numpy as np

class RunningMetric(object):
    def __init__(self, metric_type, n_classes =None):
        self._metric_type = metric_type
        if metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if metric_type == 'IOU':
            if n_classes is None:
                print('ERROR: n_classes is needed for IOU')
            self.num_updates = 0.0
            self._n_classes = n_classes
            self.confusion_matrix = np.zeros((n_classes, n_classes))
        if metric_type == 'MSE':
            self.sum = 0.0
            self.num = 0.0
        if metric_type == 'SPCC':
            self.rs = 0.0
            self.num = 0.0
        if metric_type == 'PCC':
            self.rs = 0.0
            self.num = 0.0


    def reset(self):
        if self._metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'L1':
            self.l1 = 0.0
            self.num_updates = 0.0
        if self._metric_type == 'IOU':
            self.num_updates = 0.0
            self.confusion_matrix = np.zeros((self._n_classes, self._n_classes))
        if self._metric_type == 'MSE' :
            self.sum = 0.0
            self.num = 0.0
        if self.metric_type == 'SPCC':
            self.rs = 0.0
            self.num = 0.0
        if self.metric_type == 'PCC':
            self.rs = 0.0
            self.num = 0.0


    def _fast_hist(self, pred, gt):
        mask = (gt >= 0) & (gt < self._n_classes)
        hist = np.bincount(
            self._n_classes * gt[mask].astype(int) +
            pred[mask], minlength=self._n_classes**2).reshape(self._n_classes, self._n_classes)
        return hist

    def rank_correlation(att_map, att_gd):
        """
        Function that measures Spearman’s correlation coefficient between target and output:
        """
        n = att_map.shape[1]
        upper = 6 * np.sum(np.square(att_gd - att_map), axis=-1)
        down = n * (np.square(n) - 1)
        return np.mean(1 - (upper / down))

    def update(self, pred, gt):
        if self._metric_type == 'ACC':
            predictions = pred.data.max(1, keepdim=True)[1]
            self.accuracy += (predictions.eq(gt.data.view_as(predictions)).cpu().sum()) 
            self.num_updates += predictions.shape[0]
    
        if self._metric_type == 'L1':
            _gt = gt.data.cpu().numpy()
            _pred = pred.data.cpu().numpy()
            gti = _gt.astype(np.int32)
            mask = gti!=250
            if np.sum(mask) < 1:
                return
            self.l1 += np.sum( np.abs(gti[mask] - _pred.astype(np.int32)[mask]) ) 
            self.num_updates += np.sum(mask)

        if self._metric_type == 'IOU':
            _pred = pred.data.max(1)[1].cpu().numpy()
            _gt = gt.data.cpu().numpy()
            for lt, lp in zip(_pred, _gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
        if self._metric_type == 'MSE' :
            self.sum += np.sum(np.power((gt.data.cpu().numpy().reshape(-1, 1) - pred.data.cpu().numpy().reshape(-1, 1)), 2))
            self.num += pred.shape[0]

        if self._metric_type == 'SPCC' :
            # self.rs += (gt.data.cpu().numpy().reshape(-1, 1)).corr((pred.data.cpu().numpy().reshape(-1, 1)), method='spearman')
            self.rs += self.rank_correlation((gt.data.cpu().numpy().reshape(-1, 1)), (pred.data.cpu().numpy().reshape(-1, 1)))
            self.num += pred.shape[0]

        if self._metric_type == 'PCC' :
            # self.rs += (gt.data.cpu().numpy().reshape(-1, 1)).corr((pred.data.cpu().numpy().reshape(-1, 1)), method='pearson')
            self.rs += np.corrcoef((gt.data.cpu().numpy().reshape(-1, 1)), (pred.data.cpu().numpy().reshape(-1, 1)))
            self.num += pred.shape[0]
        
    def get_result(self):
        if self._metric_type == 'ACC':
            return {'acc': self.accuracy/self.num_updates}
        if self._metric_type == 'L1':
            return {'l1': self.l1/self.num_updates}
        if self._metric_type == 'IOU':
            acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
            acc_cls = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(axis=1)
            acc_cls = np.nanmean(acc_cls)
            iou = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)) 
            mean_iou = np.nanmean(iou)
            return {'micro_acc': acc, 'macro_acc':acc_cls, 'mIOU': mean_iou}
        if self._metric_type == 'MSE' :
            return {'mse': self.sum / self.num}
        if self._metric_type == 'SPCC' :
            return {'spcc': self.rs / self.num}
        if self._metric_type == 'PCC' :
            return {'spcc': self.rs / self.num}



def get_metrics(params):
    met = {}
    if 'mnist' in params['dataset']:
        for t in params['tasks']:
            met[t] = RunningMetric(metric_type = 'ACC')
    if 'cityscapes' in params['dataset']:
        if 'S' in params['tasks']:
            met['S'] = RunningMetric(metric_type = 'IOU', n_classes=19)
        if 'I' in params['tasks']:
            met['I'] = RunningMetric(metric_type = 'L1')
        if 'D' in params['tasks']:
            met['D'] = RunningMetric(metric_type = 'L1')
    if 'celeba' in params['dataset']:
        for t in params['tasks']:
            met[t] = RunningMetric(metric_type = 'ACC')
    if 'tencent' in params['dataset']:
        for t in params['tasks']:
            met[t] = RunningMetric(metric_type = 'MSE')
    return met