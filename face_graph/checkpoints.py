# checkpoints.py

import os
import torch
from torch import nn
import torch.optim as optim
import pickle

import pdb

class Checkpoints:
    def __init__(self, args):
        self.args = args
        self.dir_save = args.save_dir
        self.model_filename = args.resume
        self.save_results = args.save_results
        self.cuda = args.cuda

        if self.save_results and not os.path.isdir(self.dir_save):
            os.makedirs(self.dir_save)

    def latest(self, name):
        if name == 'resume':
            return self.model_filename

    def save(self, acc, models, epoch):
        keys = list(models)
        assert(len(keys) == 3)
        filename = '%s/model_epoch_%d_%.6f.pth.tar' % (self.dir_save, epoch, acc)

        checkpoint = {}
        
        # save model
        keys = list(models['model'])
        for key in keys:
            newkey = 'model_'+key
            checkpoint[newkey] = models['model'][key].state_dict()

        # save criterion
        keys = list(models['loss'])
        for key in keys:
            newkey = 'loss_'+key
            checkpoint[newkey] = models['loss'][key].state_dict()
            
        # save optimizer
        keys = list(models['optimizer'])
        for key in keys:
            newkey = 'optimizer_'+key
            checkpoint[newkey] = models['optimizer'][key].state_dict()

        torch.save(checkpoint, filename)

    def load(self, models, filename, old=False):
        if old:
            # filename_model = os.path.join(filename, 'model_epoch_25_final_0.997500.pth')
            # filename_loss = os.path.join(filename, 'loss_epoch_25_final_0.997500.pth')
            # filename_model = os.path.join(filename, 'model_epoch_82_final_36.700000.pth')
            # filename_loss = os.path.join(filename, 'loss_epoch_82_final_36.700000.pkl')
            # filename_model = os.path.join(filename, 'model_epoch_200_final_90.040000.pth')
            # filename_loss = os.path.join(filename, 'loss_epoch_200_final_90.040000.pkl')

            
            filename_model = filename
            filename_loss = filename

            key_model = list(models['model'])[0]
            model = models['model'][key_model]
            if os.path.isfile(filename_model) and os.path.isfile(filename_loss):
                print("=> loading checkpoint '{}'".format(filename))
                if filename_model.endswith('pth'):
                    state_dict = torch.load(filename_model)
                    saved_params = list(state_dict)
                    
                    update_dict = {}
                    model_params = list(model.state_dict())

                    j = 0
                    for i,key in enumerate(model_params):
                        if key.endswith('num_batches_tracked'):
                            update_dict[key] = model.state_dict()[key]
                        else:
                            update_dict[key] = state_dict[saved_params[j]]
                            j += 1
                    models['model'][key_model].load_state_dict(update_dict)
                elif filename_model.endswith('pkl'):
                    with open(filename_model, 'rb') as f:
                        saved_model = pickle.load(f)
                    keys = list(saved_model)
                    for key in keys:
                        models['model'][key].load_state_dict(saved_model[key], strict=False)
                
                # if filename_loss.endswith('pkl'):
                #     with open(filename_loss, 'rb') as f:
                #         saved_loss = pickle.load(f)
                #     keys = list(saved_loss)
                #     for key in keys:
                #         models['loss'][key].load_state_dict(saved_loss[key], strict=False) #.state_dict()
                # elif filename_loss.endswith('pth'):
                #     models['loss'].load_state_dict(torch.load(filename_loss), strict=False)

                return models
        
        else:
            if os.path.isfile(filename):
                print("=> loading checkpoint '{}'".format(filename))
                if self.cuda:
                    checkpoint = torch.load(filename)
                else:
                    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
                
                keys_check = list(checkpoint)
                for key_check in keys_check:
                    if key_check.startswith('model'):
                        key = key_check.split('_')[1]
                        models['model'][key].load_state_dict(checkpoint[key_check])
                    elif key_check.startswith('loss'):
                        key = key_check.split('_')[1]
                        models['loss'][key].load_state_dict(checkpoint[key_check])
                    elif key_check.startswith('optimizer'):
                        key = key_check.split('_')[1]
                        models['optimizer'][key].load_state_dict(checkpoint[key_check])

                return models
            else:
                raise (Exception("=> no checkpoint found at '{}'".format(filename)))
