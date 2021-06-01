# test.py

import time
import plugins
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import h5py

import pdb

class Tester:
    def __init__(self, args, model, criterion, evaluation):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation

        self.nepochs = args.nepochs

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger.txt',
            args.save_results
        )
        params_loss = ['Test_Result']
        self.log_loss.register(params_loss)

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            params_loss[0]: {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # display training progress
        self.print_formatter = 'Test [%d/%d]] '
        for item in [params_loss[0]]:
            self.print_formatter += item + " %.4f "

        self.losses = {params_loss[0]:0.0}
        # self.binage = torch.Tensor([19,37.5,52.5,77])

    def model_eval(self):
        keys = list(self.model)
        for key in keys:
            self.model[key].eval()

    def test(self, epoch, dataloader):
        dataloader = dataloader['test']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to eval mode
        self.model_eval()

        end = time.time()

        features = []
        labels = []

        # extract query features
        for i, (inputs,input_labels,attrs,fmetas) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end
            end = time.time()

            ############################
            # Evaluate Network
            ############################

            if self.args.cuda:
                inputs = inputs.cuda()

            embeddings = self.model['face'](inputs)

            feat_time = time.time() - end
            
            features.append(embeddings.data.cpu().numpy())
            labels.append(input_labels.data.numpy())

            torch.sum(embeddings).backward()

        labels = np.concatenate(labels, axis=0)
        features = np.concatenate(features, axis=0)
        results,std,acc_dict = self.evaluation['face'](features)

        self.losses[list(self.losses)[0]] = results
        batch_size = 1
        self.monitor.update(self.losses, batch_size)

        # print batch progress
        print(self.print_formatter % tuple(
            [epoch + 1, self.nepochs] +
            [results]))
            
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # np.savez('/research/prip-gongsixu/codes/biasface/results/model_analysis/result_agel20.npz',
        #     preds=preds.cpu().numpy(), labels=labels.cpu().numpy())

        return results,std,acc_dict

    def extract_features(self, dataloader, n_img, n_class):
        dataloader = dataloader['test']
        self.model_eval()

        end = time.time()

        subdir = os.path.dirname(self.args.feat_savepath)
        if os.path.isdir(subdir) is False:
            os.makedirs(subdir)
        
        ######### save as h5py ###########
        if self.args.feat_savepath.endswith('hdf5'):
            dt_feat = h5py.special_dtype(vlen=np.dtype('float32'))
            dt_label = h5py.special_dtype(vlen=np.dtype('int64'))
            f_h5py = h5py.File(self.args.feat_savepath, 'w')        
            f_h5py.create_dataset('feat', shape=(n_img,), dtype=dt_feat)
            f_h5py.create_dataset('label', shape=(n_img,), dtype=dt_label)
        ######### save as h5py ###########
        
        # extract features
        for j, (inputs,testlabels,attrs,fmetas) in enumerate(dataloader):

            if j == 0:
                batch_size = inputs.size(0)

            if self.args.cuda:
                inputs = inputs.cuda()

            self.model['face'].zero_grad()
            outputs = self.model['face'](inputs)

            torch.sum(outputs).backward()

            ######### save as h5py ###########
            if self.args.feat_savepath.endswith('hdf5'):
                embeddings = outputs.data.cpu().numpy()
                labels = testlabels.numpy()
                for i in range(embeddings.shape[0]):
                    index = i+j*batch_size
                    label = int(labels[i])
                    f_h5py['feat'][index] = embeddings[i,:]
                    f_h5py['label'][index] = [labels[i]]
            ######### save as h5py ###########

            ######### save as numpy zip ###########
            if self.args.feat_savepath.endswith('npz'):
                if j == 0:
                    embeddings = outputs.data.cpu().numpy()
                    labels = testlabels.numpy()
                else:
                    embeddings = np.concatenate((embeddings, outputs.data.cpu().numpy()), axis=0)
                    labels = np.concatenate((labels, testlabels.numpy()), axis=0)
            ######### save as numpy zip ###########

            print('batch [{}/{}] saved!'.format(j+1,len(dataloader)))

        data_time = time.time() - end
        print(data_time)

        labels = labels.reshape(-1)

        ######### save as numpy zip ###########
        if self.args.feat_savepath.endswith('npz'):
            np.savez(self.args.feat_savepath, feat=embeddings, label=labels)
        ######### save as numpy zip ###########
