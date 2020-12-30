
from models.tencent_model import TencentEncoder, TencentDecoder

def get_model(params):
    data = params['dataset']
    if 'tencent' in data:
        model = {}
        model['rep'] = TencentEncoder()
        model['rep'].cuda()
        for t in params['tasks']:
            model[t] = TencentDecoder(params['patch_size'], params['dropout_prob'])
            model[t].cuda()
        return model

