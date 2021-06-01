import networkx as nx
import dgl
import torch
import scipy.sparse as spp
import numpy as np

from dgl.data import MiniGCDataset

import matplotlib.pyplot as plt

import pdb

u = torch.tensor([0, 0, 0, 0, 0])
v = torch.tensor([1, 2, 3, 4, 5])
adj1 = spp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))

g = dgl.DGLGraph((u,v))
nx.draw(g.to_networkx(), with_labels=True)
plt.show()

filename = '/research/prip-gongsixu/codes/uniface/results/tempfeats.py'
# with open(filename, 'wb') as f:
#     x = torch.rand(5,30)
#     np.save(f, x.numpy())
with open(filename, 'rb') as f:
    x = np.load(f)
    x = torch.Tensor(x)

k = 3
knn_g = dgl.knn_graph(x, k)  # Each node has two predecessors

plt.subplot(121)
nx.draw(knn_g.to_networkx(), with_labels=True)

dist = dgl.transform.pairwise_squared_distance(x)
_,indices=torch.sort(dist, dim=1)
n_nodes = indices.size(0)
cols = np.arange(n_nodes)
cols = np.repeat(cols, k)
rows = indices.numpy()
rows = rows[:,1:k+1]
rows = np.reshape(rows, -1)
data = np.ones(rows.shape[0])
adj = spp.csr_matrix((data, (rows, cols)), shape=(n_nodes,n_nodes))
adj_g = dgl.DGLGraph(adj)

plt.subplot(122)
nx.draw(adj_g.to_networkx(), with_labels=True)

plt.show()