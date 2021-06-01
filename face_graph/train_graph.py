# train.py

import time
import plugins
import itertools

import os
import numpy as np
import scipy.sparse as spp
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv
import dgl

import pdb

class Trainer:
    def __init__(self, args, model, criterion, evaluation, writer, optimizer=None):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation
        self.writer = writer

        self.nepochs = args.nepochs

        self.lr = args.lr
        self.optim_method = args.optim_method
        self.optim_options = args.optim_options
        self.scheduler_method = args.scheduler_method

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
        params_loss = ['Model', 'LearningRate','FLoss', 'GLoss', 'Acc']
        self.log_loss.register(params_loss)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'Model': {'dtype': 'str'},
            'LearningRate': {'dtype': 'running_mean'},
            'FLoss': {'dtype': 'running_mean'},
            'GLoss': {'dtype': 'running_mean'},
            'Acc': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # display training progress
        self.print_formatter = 'Train [%d/%d][%d/%d] '
        for i,item in enumerate(params_loss):
            if i == 0:
                self.print_formatter += item + " %s "
            else:
                self.print_formatter += item + " %.4f "
        # self.print_formatter += "Scale %.4f "

        self.losses = {}
        self.losses['GLoss'] = 999
        self.losses['FLoss'] = 999

    def model_train(self):
        keys = list(self.model)
        for key in keys:
            self.model[key].train()

    def train(self, epoch, dataloader):
        dataloader = dataloader['train']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to train mode
        self.model_train()

        end = time.time()

        loss_min = 999
        stored_models = {}

        g_list = []
        label_list = []
        feat_list = []
        id_list = []
        count_graph = 0

        for i, (inputs, labels, attrs, fmetas) in enumerate(dataloader):
            if i == 17144:
                break

            # keeps track of data loading time
            data_time = time.time() - end

            inputs = inputs.view(-1,inputs.size(-3),inputs.size(-2),inputs.size(-1))
            labels = labels.view(-1)

            # show images on tensorboard
            img_grid = torchvision.utils.make_grid(inputs, nrow = 5, normalize=True)
            # matplotlib_imshow(i, img_grid, one_channel=False)
            self.writer.add_image('four_fashion_mnist_images', img_grid)

            ############################
            # Update network
            ############################
            
            if self.args.cuda:
                inputs = inputs.cuda()

            feats = self.model['face'](inputs)
            feat_list.append(feats)
            id_list.append(labels)

            graphs, g_labels, ndiff = get_graph(feats, labels, self.args.nsubjects, self.args.nsamples)
            # if ndiff < self.args.nsamples:
            #     continue

            g_list.extend(graphs)
            label_list.extend(g_labels)
            count_graph += self.args.ngraphs

            if count_graph == self.args.ngraphs:
                graphs = dgl.batch(g_list)
                labels = torch.Tensor(label_list)
                graphs = graphs.to('cuda:3')
                adv_labels = torch.ones(labels.size(0),2)
                adv_labels *= 0.5
                
                feats = torch.cat(feat_list, dim=0)
                ids = torch.cat(id_list, dim=0)

                if self.args.cuda:
                    labels = labels.cuda()
                    adv_labels = adv_labels.cuda()
                    ids = ids.to(feats.device)

                prediction = self.model['graph'](graphs)

                # update graph classifier
                if float(epoch) % 3.0 == 0:
                    loss = self.criterion['graph'](prediction, labels)
                    self.optimizer['graph'].zero_grad()
                    loss.backward()
                    self.optimizer['graph'].step()
                    cur_lr = self.optimizer['graph'].param_groups[0]['lr']
                    self.losses['Model'] = 'Graph'
                    self.losses['GLoss'] = loss.item()

                # update face model
                else:
                    face_loss = self.criterion['face'](feats, ids)
                    adv_loss = self.criterion['adv'](prediction, adv_labels)
                    loss = face_loss+(1-self.monitor.getvalues()['GLoss'])*adv_loss              
                    self.optimizer['face'].zero_grad()
                    loss.backward()
                    self.optimizer['face'].step()
                    cur_lr = self.optimizer['face'].param_groups[0]['lr']
                    self.losses['Model'] = 'Face'
                    self.losses['FLoss'] = loss.item()
                
                g_list = []
                label_list = []
                feat_list = []
                id_list = []
                count_graph = 0

                batch_size = labels.size(0)
                _, pred = prediction.topk(1, 1, True, True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                correct_k = correct[:1].view(-1).float().sum(0)
                res = correct_k.mul_(100.0 / batch_size)
                self.losses['Acc'] = res               
                self.losses['LearningRate'] = cur_lr                
                self.monitor.update(self.losses, batch_size)

                # print batch progress
                print(self.print_formatter % tuple(
                    [epoch + 1, self.nepochs, i+1, len(dataloader)] +
                    [self.losses[key] for key in self.params_monitor]))
                
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # update the learning rate
        if self.scheduler_method is not None:
            if self.scheduler_method == 'ReduceLROnPlateau':
                self.scheduler.step(loss['Loss'])
            elif self.scheduler_method == 'Customer':
                self.update_lr('face', epoch)
                self.update_lr('graph', epoch)
            else:
                self.scheduler.step()

        return self.monitor.getvalues('FLoss')

    def update_lr(self, key, epoch):
        if epoch in self.args.lr_schedule[key]: 
            self.lr[key] *= 0.1
            for param_group in self.optimizer[key].param_groups:
                param_group['lr'] = self.lr[key]

def get_graph(feat, label, nsubjects, nsamples):
    k = nsamples - 1 # number of neighbors
    n_nodes = nsubjects*nsamples
    ngraphs = int(feat.size(0)/n_nodes)
    data = np.ones(n_nodes*k)
    
    g_list = []
    labels = []
    ndiff = 0
    for i in range(ngraphs):
        ndata = feat[i*n_nodes:(i+1)*n_nodes,:]
        ndata = ndata.cpu()
        ndata = F.normalize(ndata)
        
        # target graph
        cols = np.arange(n_nodes)
        cols = np.repeat(cols, k)
        rows = np.arange(n_nodes)
        rows = np.reshape(rows, (-1,nsamples))
        rows = np.repeat(rows, nsamples, axis=0)
        to_del = []
        for j in range(nsubjects):
            for t in range(nsamples):
                to_del.append(j*nsamples**2 + t*(nsamples+1))
        rows = np.delete(rows, to_del, None)
        coordinate1 = list(zip(rows, cols))
        adj = spp.csr_matrix((data, (rows, cols)), shape=(n_nodes,n_nodes))
        graph = dgl.DGLGraph(adj)
        graph.ndata['w'] = ndata
        g_list.append(graph)
        labels.append(1)

        # plt.subplot(121)
        # nx.draw(graph.to_networkx(), with_labels=True)

        # oberseved graph
        dist = dgl.transform.pairwise_squared_distance(ndata)
        _,indices = torch.sort(dist, dim=1)
        cols = np.arange(n_nodes)
        cols = np.repeat(cols, k)
        rows = indices.cpu().numpy()
        rows = rows[:,1:k+1]
        rows = np.reshape(rows, -1)
        coordinate2 = list(zip(rows, cols))
        adj = spp.csr_matrix((data, (rows, cols)), shape=(n_nodes,n_nodes))
        graph = dgl.DGLGraph(adj)
        graph.ndata['w'] = ndata
        g_list.append(graph)
        labels.append(0)

        # plt.subplot(122)
        # nx.draw(graph.to_networkx(), with_labels=True)

        diff = list(set(coordinate1)-set(coordinate2))
        ndiff += len(diff)

    return g_list, labels, ndiff

def matplotlib_imshow(ind, img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()