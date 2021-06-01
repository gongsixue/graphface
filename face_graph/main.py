# main.py

import os
import sys
import traceback
import random
import config
import utils
from model import Model
from dataloader import Dataloader
from checkpoints import Checkpoints
import torch

from torch.utils.tensorboard import SummaryWriter

args, config_file = config.parse_args()
# Data Loading    
if args.train == 'face_cls':
    from test_cls import Tester
    from train_cls import Trainer

if args.train == 'face_graph':
    from test_cls import Tester
    from train_graph import Trainer

if args.dataset_train == 'ClassSamplesDataLoader':
    from train_classload import Trainer


def main():
    # parse the arguments
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.save_results:
        utils.saveargs(args, config_file)

    # initialize the checkpoint class
    checkpoints = Checkpoints(args)

    # Create Model
    models = Model(args)
    model, model_dict, evaluation = models.setup(checkpoints)

    print('Model:\n\t{model}\nTotal params:\n\t{npar:.2f}M'.format(
          model=args.model_type['face'],
          npar=sum(p.numel() for p in model['face'].parameters()) / 1000000.0))

    writer = SummaryWriter(args.tblog_dir)

    # The trainer handles the training loop
    trainer = Trainer(args, model, model_dict['loss'], evaluation, writer, model_dict['optimizer'])
    # The trainer handles the evaluation on validation set
    tester = Tester(args, model, model_dict['loss'], evaluation)

    test_freq = 3

    dataloader = Dataloader(args)

    if args.extract_feat:
        loaders  = dataloader.create(flag='Test')
        tester.extract_features(loaders, args.n_img, len(dataloader.dataset_test.classname))
    elif args.just_test:
        loaders  = dataloader.create(flag='Test')
        avg,std,acc_dict = tester.test(args.epoch_number, loaders)
        # print(avg)
        print(std)
        # print(acc_dict)
    else:
        loaders  = dataloader.create()
        if args.dataset_train == 'ClassSamplesDataLoader':
            loaders['train'] = dataloader.dataset_train
        # start training !!!
        acc_best = 0
        loss_best = 999
        stored_models = {}

        for epoch in range(args.nepochs-args.epoch_number):
            epoch += args.epoch_number
            print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))

            # train for a single epoch
            # loss_train = 3.0
            loss_train = trainer.train(epoch, loaders)
            if float(epoch) % test_freq != 0:
                acc_test,_,_ = tester.test(epoch, loaders)
                acc_best = acc_test

            if loss_best > loss_train:
                loss_best = loss_train
                if args.save_results:
                    stored_models['model'] = model
                    stored_models['loss'] = trainer.criterion
                    stored_models['optimizer'] = trainer.optimizer
                    checkpoints.save(acc_best, stored_models, epoch)


if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        utils.cleanup()
