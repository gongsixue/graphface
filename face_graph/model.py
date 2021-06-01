# model.py

import math

import evaluate

import losses
import models
from torch import nn
import torch.optim as optim

import pdb

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.bias.data.zero_()


class Model:
    def __init__(self, args):
        self.args = args

        self.ngpu = args.ngpu
        self.cuda = args.cuda
        self.model_type = args.model_type
        self.model_options = args.model_options
        self.loss_type = args.loss_type
        self.loss_options = args.loss_options
        self.evaluation_type = args.evaluation_type
        self.evaluation_options = args.evaluation_options

    def setup(self, checkpoints):
        model = {}
        keys_model = list(self.model_type)
        for key in keys_model:
            model[key] = getattr(models, self.model_type[key])(**self.model_options[key])
        
        criterion = {}
        keys_loss = list(self.loss_type)
        for key in keys_loss:
            criterion[key] = getattr(losses, self.loss_type[key])(**self.loss_options[key])
        
        evaluation = {}
        key = 'face'
        evaluation[key] = getattr(evaluate, self.evaluation_type[key])(
            **self.evaluation_options[key])

        if self.cuda:
            for key in keys_model:
                if key == 'face':
                    model[key] = nn.DataParallel(model[key], device_ids=list(range(self.ngpu)))
                    model[key] = model[key].cuda()
                elif key == 'graph':
                    model['graph'] = model['graph'].to('cuda:3')
            for key in keys_loss:
                criterion[key] = criterion[key].cuda()

        model_dict = {}
        model_dict['model'] = model
        model_dict['loss'] = criterion
        model_dict['optimizer'] = {}
        lr = self.args.lr
        module_list = [model['face'], criterion['face']]
        module_list = nn.ModuleList(module_list)
        model_dict['optimizer']['face'] = getattr(optim, self.args.optim_method)(
            module_list.parameters(), lr=lr['face'], **self.args.optim_options)
        model_dict['optimizer']['graph'] = getattr(optim, self.args.optim_method)(
            model['graph'].parameters(), lr=lr['graph'], **self.args.optim_options)

        if checkpoints.latest('resume') is None:
            pass
            # model.apply(weights_init)
        else:
            model_dict = checkpoints.load(model_dict, checkpoints.latest('resume'), old=self.args.old)

        return model, model_dict, evaluation
