
from models.tencent_model import TencentEncoder, TencentDecoder

def get_model(params):
    data = params['dataset']
    if 'tencent' in data:
        model = {}
        # model['rep'] = ResNet(BasicBlock, [2,2,2,2])
        model['rep'] = TencentEncoder()
        model['rep'].cuda()
        for t in params['tasks']:
            model[t] = TencentDecoder(params['patch_size'])
            model[t].cuda()
        return model

