# train.py

import time
import plugins
import itertools

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import pdb

class Trainer:
    def __init__(self, args, model, criterion, evaluation, optimizer=None):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation

        self.nepochs = args.nepochs

        self.lr = args.lr
        self.optim_method = args.optim_method
        self.optim_options = args.optim_options
        self.scheduler_method = args.scheduler_method

        if optimizer is None:
            self.module_list = nn.ModuleList([self.model, self.criterion])
            self.optimizer = getattr(optim, self.optim_method)(
                self.module_list.parameters(), lr=self.lr, **self.optim_options)
        else:
            self.optimizer = optimizer
        if self.scheduler_method is not None:
            if self.scheduler_method != 'Customer':
                self.scheduler = getattr(optim.lr_scheduler, self.scheduler_method)(
                    self.optimizer, **args.scheduler_options)

        # logging training
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TrainLogger.txt',
            args.save_results
        )
        params_loss = ['LearningRate','Loss']
        self.log_loss.register(params_loss)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'LearningRate': {'dtype': 'running_mean'},
            'Loss': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # display training progress
        self.print_formatter = 'Train [%d/%d][%d/%d] '
        for item in params_loss:
            self.print_formatter += item + " %.4f "
        # self.print_formatter += "Scale %.4f "

        self.losses = {}
        self.binage = torch.Tensor([10,22.5,27.5,32.5,37.5,42.5,47.5,55,75])

    def model_train(self):
        keys = list(self.model)
        for key in keys:
            self.model[key].train()

    def train(self, epoch, dataloader, checkpoints, acc_best):
        dataloader = dataloader['train']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to train mode
        self.model_train()

        end = time.time()

        loss_min = 999
        stored_models = {}

        for i, (inputs, labels, attrs, fmetas) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end

            ############################
            # Update network
            ############################

            batch_size = inputs.size(0)
            
            if self.args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
            labels = labels.view(-1)

            outputs = self.model['face'](inputs)

            loss = self.criterion['face'](outputs, labels)

            ########## evaluation ###########
            # acc_batch = self.evaluation(outputs)
            # acc += acc_batch
            ########## evaluation ###########

            self.optimizer['face'].zero_grad()
            loss.backward()
            self.optimizer['face'].step()

            self.losses['Loss'] = loss.item()
            for param_group in self.optimizer['face'].param_groups:
                self.cur_lr = param_group['lr']
            self.losses['LearningRate'] = self.cur_lr
            self.monitor.update(self.losses, batch_size)

            # print batch progress
            print(self.print_formatter % tuple(
                [epoch + 1, self.nepochs, i+1, len(dataloader)] +
                [self.losses[key] for key in self.params_monitor]))

            # if i%10000 == 0:
            #     if self.losses['Loss'] < loss_min:
            #         loss_min == self.losses['Loss']
            #         stored_models['model'] = self.model
            #         stored_models['loss'] = self.criterion
            #         stored_models['optimizer'] = self.optimizer
            #         checkpoints.save(acc_best, stored_models, epoch, i, True)
        
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # print epoch progress
        # print(self.print_formatter % tuple(
        #     [epoch + 1, self.nepochs, i+1, len(dataloader)] +
        #     [loss[key] for key in self.params_monitor]))

        # update the learning rate
        if self.scheduler_method is not None:
            if self.scheduler_method == 'ReduceLROnPlateau':
                self.scheduler.step(loss['Loss'])
            elif self.scheduler_method == 'Customer':
                if epoch in self.args.lr_schedule: 
                    self.lr *= 0.1
                    for param_group in self.optimizer['face'].param_groups:
                        param_group['lr'] = self.lr
            else:
                self.scheduler.step()

        return self.monitor.getvalues('Loss')
