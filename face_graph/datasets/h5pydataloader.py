# filelist.py

import os
import io
import math
import utils as utils
import torch.utils.data as data
import datasets.loaders as loaders
from PIL import Image
import h5py

import torch
import random

import pdb

class H5pyLoader(data.Dataset):
    def __init__(self, ifile, root=None,
        transform=None, loader='loader_image'):

        self.root = root
        self.transform = transform
        if loader is not None:
            self.loader = getattr(loaders, loader)

        self.f_h5py = h5py.File(ifile[0], 'r')
        
        if ifile[1].endswith('txt'):
            lines = utils.readtextfile(ifile[1])
            imagelist = []
            for x in lines:
                x = x.rstrip('\n')
                filename = os.path.splitext(os.path.basename(x))[0]
                labelname = os.path.basename(os.path.dirname(x))
                temp = [os.path.join(labelname, filename + '.jpg')]
                temp.append(labelname)
                imagelist.append(temp)

        labellist = [x[1] for x in imagelist]

        self.images = imagelist
        self.classname = labellist
        self.classname = list(set(self.classname))
        self.classname.sort()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if len(self.images) > 0:
            if self.root is not None:
                path = os.path.join(self.root,self.images[index][0])
            else:
                path = self.images[index][0]

            label = self.classname.index(self.images[index][1])
            fmeta = path

            im_bytes = self.f_h5py['images'][index]
            image = Image.open(io.BytesIO(im_bytes))

            if self.transform is not None:
                image = self.transform(image)

        else:
            image = []
            label = None
            fmeta = None        

        return image, label, label, fmeta

class H5py_ClassLoader(data.Dataset):
    def __init__(self, ifile_image, ifile_index, nsamples, transform=None):

        self.transform = transform
        
        self.f_h5py = h5py.File(ifile_image, 'r')

        self.index_dict = h5py.File(ifile_index, 'r')
        self.classname = list(self.index_dict)
        self.classname.sort()

        self.nsamples = nsamples

    def __len__(self):
        return len(self.classname)

    def __getitem__(self, index):
        key = self.classname[index]
        label = self.classname.index(key)
        indices = [x[0] for x in self.index_dict[key]]
        nfeat = len(indices)
        if nfeat < self.nsamples:
            num = int(self.nsamples/nfeat)
            det = int(self.nsamples%nfeat)
            cur_indices = random.sample(indices,det)
            for i in range(num):
                cur_indices.extend(indices)
        else:
            cur_indices = random.sample(indices,self.nsamples)

        images = []
        fmeta = []
        for i in cur_indices:
            im_bytes = self.f_h5py['images'][i]
            image = Image.open(io.BytesIO(im_bytes))
            image = self.transform(image)
            images.append(torch.unsqueeze(image, 0))
            fmeta.append(self.f_h5py['paths'][i])
        images = torch.cat(images)

        label = torch.Tensor([label]*self.nsamples)
        cur_indices = torch.Tensor(cur_indices)
        fmeta = [self.classname[index]]*self.nsamples

        return images, label, cur_indices, fmeta
