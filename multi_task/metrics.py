# Adapted from: https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py

import numpy as np
from scipy import stats

class RunningMetric(object):
    def __init__(self, metric_type, n_classes =None):
        self._metric_type = metric_type
        if self._metric_type == 'MULTI':
            # MSE SPCC PCC ACC
            self.gt = []
            self.pred = []

    def reset(self):
        if self._metric_type == 'MULTI':
            # MSE SPCC PCC ACC_DIS
            self.gt = []
            self.pred = []

    def calculate_score(self, dis):
        weights = np.array([1, 2, 3, 4 ,5])
        return np.sum(dis * weights, axis=1)

    def MSE(self, pred, gt):
        return np.square(pred - gt).mean()

    def accuracy(self, pred, gt):
        pred_ge_3 = pred >= 3
        gt_ge_3 = gt >= 3
        return np.sum(pred_ge_3 ==  gt_ge_3) / pred.shape[0]


    def update(self, pred, gt):
        if self._metric_type == 'MULTI':
            gt_score = self.calculate_score(gt.data.cpu().numpy().reshape(-1,5)).tolist()
            self.gt += gt_score
            pred_score = self.calculate_score(pred.data.cpu().numpy().reshape(-1,5)).tolist()
            self.pred += pred_score

        
    def get_result(self):
        if self._metric_type == 'MULTI':
            pred = np.array(self.pred)
            gt = np.array(self.gt)
            return {
                'mse': self.MSE(pred, gt),
                'spcc': stats.spearmanr(pred, gt)[0],
                'plcc': stats.pearsonr(pred, gt)[0],
                'acc': self.accuracy(pred, gt)
            }

def get_metrics(params):
    met = {}
    if 'tencent' in params['dataset']:
        for t in params['tasks']:
            met[t] = RunningMetric(metric_type = 'MULTI')
    return met