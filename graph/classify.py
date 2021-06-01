from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv
import dgl

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import scipy.sparse as spp
import h5py

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networkx.drawing.nx_agraph import write_dot, graphviz_layout

import pdb

class FeatLoader(data.Dataset):
    def __init__(self, ifile_feat, ifile_index, nsamples):
        # h5py
        # data = h5py.File(ifile_feat, 'r')
        # self.feats = data['feat']       
        # self.index_dict = h5py.File(ifile_index, 'r')
        # self.classname = list(self.index_dict)

        # npz
        data = np.load(ifile_feat)
        feats = data['feat']
        labels = data['label']
        self.classname = list(set(list(labels)))
        self.featdict = {}
        for i,lab in enumerate(labels):
            label = self.classname.index(lab)
            if label not in self.featdict:
                self.featdict[label] = [feats[i,:]]
            else:
                self.featdict[label].append(feats[i,:])

        self.nsamples = nsamples

    def __len__(self):
        return len(self.classname)

    def __getitem__(self, index):
        key = self.classname[index]
        label = self.classname.index(key)

        # h5py
        # indices = [x[0] for x in self.index_dict[key]]
        # cur_feat = [x for x in self.feats[indices]]
        # nfeat = len(indices)

        # npz
        cur_feat = self.featdict[key]
        nfeat = len(cur_feat)
        
        if nfeat < self.nsamples:
            num = int(self.nsamples/nfeat)
            det = int(self.nsamples%nfeat)
            subfeats = random.sample(cur_feat,det)
            for i in range(num):
                subfeats.extend(cur_feat)
        else:
            subfeats = random.sample(cur_feat,self.nsamples)
        feat = torch.Tensor(subfeats)
        label = torch.Tensor([label]*self.nsamples)

        return feat, label

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.ndata['w']
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

def get_graph(feat, label, nsubjects, nsamples):
    ngraphs = int(feat.size(0)/nsubjects)
    k = nsamples - 1 # number of neighbors
    n_nodes = nsubjects*nsamples
    data = np.ones(n_nodes*k)

    colors = []
    for i in range(25):
        if int(i/5) == 0:
            colors.append([.6,.6,.6])
        if int(i/5) == 1:
            colors.append([0,.9,.5])
        if int(i/5) == 2:
            colors.append([.9,.5,0])
        if int(i/5) == 3:
            colors.append([1,.5,.5])
        if int(i/5) == 4:
            colors.append([.2,.7,.9])
    
    g_list = []
    labels = []
    ndiff = 0
    for i in range(ngraphs):
        ndata = feat[i*nsubjects:(i+1)*nsubjects]
        ndata = ndata.view(-1,ndata.size(-1))
        
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

        # # ax1 = plt.subplot(121)
        # graph = graph.to_networkx()
        # # pos = nx.kamada_kawai_layout(graph)
        # pos = graphviz_layout(graph)
        # nx.draw(graph, pos, with_labels=True, node_color=colors)
        # plt.show()

        # oberseved graph
        dist = dgl.transform.pairwise_squared_distance(ndata)
        _,indices = torch.sort(dist, dim=1)
        cols = np.arange(n_nodes)
        cols = np.repeat(cols, k)
        rows = indices.numpy()
        rows = rows[:,1:k+1]
        rows = np.reshape(rows, -1)
        coordinate2 = list(zip(rows, cols))
        adj = spp.csr_matrix((data, (rows, cols)), shape=(n_nodes,n_nodes))
        graph = dgl.DGLGraph(adj)
        graph.ndata['w'] = ndata
        g_list.append(graph)
        labels.append(0)

        # ax2 = plt.subplot(122)
        graph = graph.to_networkx()
        # pos = nx.kamada_kawai_layout(graph)
        pos = graphviz_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color=colors)
        plt.show()

        diff = list(set(coordinate1)-set(coordinate2))
        ndiff += len(diff)
        # if len(diff) > 4:
        #     plt.show()
        #     pdb.set_trace()
        # ax1.clear()
        # ax2.clear()

    # graphs = dgl.batch(g_list)
    # labels = torch.Tensor(labels)
    return g_list, labels, ndiff

def train(ifile_feat, ifile_index, ngraphs, 
    nsamples, nsubjects, nthreads, nclasses, npu, savepath):
    dataset = FeatLoader(ifile_feat, ifile_index, nsamples)
    dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=nsubjects,
                    num_workers=int(nthreads),
                    shuffle=True,pin_memory=True
                )

    # Create model
    model = Classifier(512, 128, nclasses)
    model = nn.DataParallel(model, device_ids=list(range(ngpu)))

    loss_func = nn.CrossEntropyLoss()
    loss_func = loss_func.cuda()

    lr = 0.01
    lr_schedule = [15, 25, 30]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    loss_best = 99.9
    for epoch in range(50):
        epoch_loss = 0
        g_list = []
        label_list = []
        count_graph = 0
        for iter, (feat, label) in enumerate(dataloader):
            # create graphs
            graphs, labels, ndiff = get_graph(feat, label, nsubjects, nsamples)
            if ndiff < nsamples:
                continue
            
            g_list.extend(graphs)
            label_list.extend(labels)
            count_graph += 1
            
            if count_graph == ngraphs:
                graphs = dgl.batch(g_list)
                labels = torch.Tensor(label_list)
                graphs = graphs.to('cuda:0')
                labels = labels.cuda()

                prediction = model(graphs)
                loss = loss_func(prediction, labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                g_list = []
                label_list = []
                count_graph = 0
        epoch_loss /= (iter + 1)
        cur_lr = optimizer.param_groups[0]['lr']
        print('Epoch {}, lr {:.5f}, loss {:.4f}'.format(epoch, cur_lr, epoch_loss))
        
        # save models
        if epoch_loss < loss_best:
            loss_best = epoch_loss
            res = test(model, ifile_feat, ifile_index, nsamples, nsubjects)

            filename = '%s/model_epoch_%d_%.4f.pth.tar' % (savepath, epoch, res)
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['loss'] = loss_func.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, filename)

        # update learning rate
        if epoch in lr_schedule:
            lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    return model

def test(model, fifile_feat, ifile_index, nsamples, nsubjects):
    ngraphs = 32*32
    dataset = FeatLoader(ifile_feat, ifile_index, nsamples)
    # total_subjects = len(list(dataset.featdict))
    dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=nsubjects,
                    num_workers=3,
                    shuffle=False,pin_memory=True
                )

    model.eval()
    topk = 1
    for i in range(1):
        g_list = []
        label_list = []
        count_graph = 0
        for iter, (feat, label) in enumerate(dataloader):
            # create graphs
            graphs, labels, ndiff = get_graph(feat, label, nsubjects, nsamples)
            if ndiff < nsamples:
                continue

            g_list.extend(graphs)
            label_list.extend(labels)
            count_graph += 1
        
            if count_graph == ngraphs:
                graphs = dgl.batch(g_list)
                labels = torch.Tensor(label_list)
                graphs = graphs.to('cuda:0')
                labels = labels.cuda()

                batch_size = labels.size(0)

                prediction = model(graphs)
                _, pred = prediction.topk(topk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                correct_k = correct[:topk].view(-1).float().sum(0)
                res = correct_k.mul_(100.0 / batch_size)
                print('Accuracy {:.4f}'.format(res))

                g_list = []
                label_list = []
                count_graph = 0

                return res

if __name__ == '__main__':
    # show_graph()
    ifile_feat = '/research/prip-gongsixu/codes/uniface/results/features/ijba/feat_arcface100_ijba.npz'
    ifile_index = '/research/prip-gongsixu/codes/uniface/results/features/msceleb/index_featdict_arcface100_msceleb.hdf5'
    savepath = '/research/prip-gongsixu/codes/uniface/results/models/graph_arcface100_test'
    ngraphs = 32
    nsubjects = 5
    nsamples = 5
    nthreads = 3
    nclasses = 2
    ngpu = 1
    model = train(ifile_feat, ifile_index, ngraphs, 
        nsamples, nsubjects, nthreads, nclasses, ngpu, savepath)
    # test(model, ifile_feat, ifile_index, nsamples, nsubjects)