import os
import torchvision.transforms as transforms

import models
import datasets

# ======================== Main Setings ====================================
log_type = 'traditional'
train = 'face_graph'
save_results = True
result_path = '/research/prip-gongsixu/codes/uniface/results/models/graph_arcface100'
extract_feat = False
just_test = False
feat_savepath = '/research/prip-gongsixu/codes/uniface/results/features/feat_arcface_lfw.npz'

# resume = None
resume = '/research/prip-gongsixu/results/models/fr_base/arc100_ms1m/Save/model_epoch_21_0.997667.pth.tar'
# resume = '/research/prip-gongsixu/results/models/fr_sota/facenet_vgg2.pth'
# resume = '/research/prip-gongsixu/results/models/fr_sota/facenet.pth'
# resume = '/research/prip-gongsixu/results/models/fr_sota/model_ir_se50.pth'
old = False

log_type = 'TensorBoard'
tblog_dir = os.path.join(result_path, 'tblog')
# ======================== Main Setings ====================================

# ======================= Data Setings =====================================
dataset_root_test = None
# dataset_root_test = '/research/prip-gongsixu/datasets/LFW/lfw_aligned_retina_112'
# dataset_root_test = '/research/prip-gongsixu/datasets/CFP/cfp_aligned_retina_112'
dataset_root_train = '/research/prip-gongsixu/datasets/RFW'

# input data size
image_height = 112
image_width = 112
image_size = (image_height, image_width)

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
# preprocessing
preprocess_train = transforms.Compose([ \
        transforms.CenterCrop(image_size), \
        transforms.Resize(image_size), \
        # transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)), \
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2), \
        # transforms.RandomApply([datasets.loaders.GaussianBlur([.1, 2.])], p=0.5), \
        transforms.RandomHorizontalFlip(), \
        # transforms.RandomVerticalFlip(), \
        # transforms.RandomRotation(10), \
        transforms.ToTensor(), \
        normalize \
    ])

preprocess_test = transforms.Compose([ \
        transforms.CenterCrop(image_size), \
        transforms.Resize(image_size), \
        # transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)), \
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2), \
        # transforms.RandomApply([datasets.loaders.GaussianBlur([.1, 2.])], p=0.5), \
        # transforms.RandomHorizontalFlip(), \
        # transforms.RandomVerticalFlip(), \
        # transforms.RandomRotation(10), \
        transforms.ToTensor(), \
        normalize \
    ])

loader_input = 'loader_image'
loader_label = 'loader_numpy'

# graph configuration
nsamples = 3
nsubjects = 5
ngraphs = 16

# dataset_train = 'CSVListLoader'
# dataset_train = 'FileListLoader'
# dataset_train = 'H5pyLoader'
dataset_train = 'H5py_ClassLoader'
# dataset_train = 'ClassSamplesDataLoader'
# input_filename_train = '/research/prip-gongsixu/datasets/RFW/attr_rfw_w1000_aligned_112.txt'
input_filename_train = ['/scratch/gongsixue/msceleb_AlignedAsArcface_images.hdf5',\
    '/research/prip-gongsixu/codes/biasface/datasets/list_faces_emore.txt']
label_filename_train = '/research/prip-gongsixu/codes/uniface/results/features/msceleb/index_featdict_arcface100_msceleb.hdf5'
# dataset_options_train = {'ifile':input_filename_train, 'root':dataset_root_train,
#                  'transform':preprocess_train, 'loader':loader_input} # H5pyLoader
# dataset_options_train = {'root':dataset_root_train, 'ifile':input_filename_train,
#                  'num_images':10, 'transform':preprocess_test, 'loader':loader_input,\
#                  'train_type':train} # ClassSamplesDataLoader
dataset_options_train = {'ifile_image':input_filename_train[0],\
    'ifile_index':label_filename_train,\
    'nsamples':nsamples, 'transform':preprocess_train}

# dataset_test = 'CSVListLoader'
dataset_test = 'FileListLoader'
# dataset_test = 'H5pyLoader'
# input_filename_test = '/research/prip-gongsixu/datasets/RFW/attr_rfw_test_aligned_112.txt'
input_filename_test = '/research/prip-gongsixu/datasets/LFW/list_lfw_aligned_retina_112.txt'
# input_filename_test = '/research/prip-gongsixu/codes/biasface/datasets/list_face_demog_groups_eccv2020.csv'
# input_filename_test = ['/scratch/gongsixue/msceleb_AlignedAsArcface_images.hdf5',\
#     '/research/prip-gongsixu/codes/biasface/datasets/list_faces_emore.txt']
# n_img = 5822653 # msceleb
n_img = 13233 # lfw

label_filename_test = None
dataset_options_test = {'ifile':input_filename_test, 'root':dataset_root_test,
                 'transform':preprocess_test, 'loader':loader_input}

save_dir = os.path.join(result_path,'Save')
logs_dir = os.path.join(result_path,'Logs')
# ======================= Data Setings =====================================

# ======================= Network Model Setings ============================
# cpu/gpu settings
cuda = True
ngpu = 4
nthreads = 1

# nclasses = 26000
# nclasses = 24500
# nclasses = 22000
# nclasses = 25845

# nclasses = 28000 # balance
# nclasses = 38737 # unbalance
# nclasses = 75460 # ms1m wo rfw
nclasses = 85742 # cleaned ms1m
# nclasses = 10575 # casia

# model_type = {'face':'incep_resnetV1'}
# model_options = {'face':{"nchannels":3,"nfeatures":128, "drop_prob":0.2}}
# model_type = {'face':'sphereface20'}
# model_options = {'face':{"nchannels":3, "nfilters":64, \
#     "ndim":512, "nclasses":nclasses, "dropout_prob":0.4, "features":False}}

model_type = {'face':'resnet_face100', 'graph':'GraphClassifier'}
model_options = {'face':{"nclasses": nclasses}, \
    'graph':{"in_dim":512, "hidden_dim":128, "n_classes":2}}

# model_type = {'face':'Backbone'}
# model_options = {'face':{"num_layers": 50, "drop_ratio": 0.6, "mode": 'ir_se'}}

# loss_type = {'face':'Classification'}
# loss_options = {'face':{"if_cuda":cuda}}
# loss_type = {'face':'Softmax'}
# loss_options = {'face':{"nfeatures":128, "nclasses":nclasses, "if_cuda":cuda}}
# loss_type = 'Regression'
# loss_options = {}
loss_type = {'face':'AM_Softmax', 'adv':'CrossEntropy', 'graph':'Classification'}
loss_options = {'face':{"nfeatures":512, "nclasses":nclasses, "s":64.0, "m":0.35}, 'adv':{},\
    'graph':{}}
# ======================= Network Model Setings ============================

# ======================= Training Settings ================================
# initialization
manual_seed = 0
nepochs = 300
epoch_number = 0

# batch
batch_size = nsubjects*ngraphs
test_batch_size = 70

# optimization
# optim_method = 'Adam'
# optim_options = {"betas": (0.9, 0.999)}
optim_method = "SGD"
optim_options = {"momentum": 0.9, "weight_decay": 5e-4}

# learning rate
lr = {'face':5e-2, 'graph':1e-1}
# scheduler_method = 'CosineAnnealingLR'
scheduler_method = 'Customer'
scheduler_options = {"T_max": nepochs, "eta_min": 1e-6}
# lr_schedule = [8,13,15]
lr_schedule = {'face':[11,19,22], 'graph':[3,9,15]}
# lr_schedule = [45000,120000,195000,270000]
# ======================= Training Settings ================================

# ======================= Evaluation Settings ==============================
# label_filename = os.path.join('/research/prip-gongsixu/results/feats/evaluation', 'list_lfwblufr.txt')
label_filename = input_filename_test

# protocol and metric
protocol = 'LFW'
# protocol = 'BLUFR'
metric = 'cosine'

# files related to protocols
# IJB
eval_dir = '/research/prip-gongsixu/results/evaluation/ijbb/sphere/cs3'
# eval_dir = '/research/prip-gongsixu/results/evaluation/ijba'
imppair_filename = os.path.join(eval_dir, 'imp_pairs.csv')
genpair_filename = os.path.join(eval_dir, 'gen_pairs.csv')
pair_index_filename={'imposter':imppair_filename,'genuine':genpair_filename}
# pair_index_filename = eval_dir
template_filename = os.path.join(eval_dir, 'temp_dict.pkl')

# LFW
pairs_filename = '/research/prip-gongsixu/results/evaluation/lfw/lfw_pairs.txt'
nfolds=10

# RFW
# pairs_filename = '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/African/African_pairs.txt'
# pairs_filename = {'African': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/African/African_pairs.txt',\
#     'Asian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Asian/Asian_pairs.txt',\
#     'Caucasian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Caucasian/Caucasian_pairs.txt',\
#     'Indian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Indian/Indian_pairs.txt'}

# features saved as npm
nimgs=None
ndim=None

evaluation_type = {'face': 'FaceVerification'}
evaluation_options = {'face':{'label_filename': label_filename,\
    'protocol': protocol, 'metric': metric,\
    'nthreads': nthreads, 'multiprocess':True,\
    'pair_index_filename': pair_index_filename,'template_filename': template_filename,\
    'pairs_filename': pairs_filename, 'nfolds': nfolds,\
    'nimgs': nimgs, 'ndim': ndim}}

# evaluation_type = 'Top1Classification'
# evaluation_type = 'BiOrdinalClassify'
# evaluation_type = 'Agergs_classification'
# evaluation_options = {}
# ======================= Evaluation Settings ==============================
