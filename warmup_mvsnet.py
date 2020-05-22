import os
import time
import pyproj
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.data_util as data_util
from utils.model_util import FlowNetS, net_init
from utils.basic import PSMNet

from utils import geo_utils

class Trainer(object):
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.n_epochs = args.epochs
        self.batch_size = args.batch_size
        self.start_epoch = args.input_epoch
        # self.D = nn.DataParallel(FlowNetS()).cuda()
        model = nn.DataParallel(PSMNet(maxdisp=49))
        self.D = model.cuda()
        self.L = nn.L1Loss().cuda()
        self.optimizer = torch.optim.Adadelta(self.D.parameters(), lr=1e-1)
        self.train_loader = train_loader
        self.test_loader = test_loader
        np.random.seed(args.seed)

    def run(self):
        # net_init(self.D)
        self.D.train()
        loss_val = float('inf')
        self.train_loss = []
        outputs = []
        cv_losses = []
        print('start training')
        for j in range(self.n_epochs-self.start_epoch):
            begin = time.time()
            train_epoch_loss = []
            test_epoch_loss = []
            for k, batch in enumerate(self.train_loader):
                #forward calculation and back propagation, X: B x P x 2 x W x H
                img_pair, pre_disp,filenames = batch['images'].float(), batch['disps'].float(), batch['names']
                # import ipdb;ipdb.set_trace()
                self.optimizer.zero_grad()
                # X = img_pair.cuda()
                # Disp = pre_disp.cuda()
                # disparity_map = self.D(X)[0][:, 0, :, :] # only left disp map

                left = img_pair[:, 0:1, :self.args.img_size, :self.args.img_size].cuda()
                right = img_pair[:, 1:2, :self.args.img_size, :self.args.img_size].cuda()
                Disp = pre_disp[:, :self.args.img_size, :self.args.img_size].cuda()
                disparity_map = self.D(left, right) # PSMNet

                loss = self.L(Disp, disparity_map)
                loss_val = loss.data.cpu().numpy()
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss_val)
                # del X,Disp,loss
                del left,right,Disp,loss
                print("Epochs %d, iteration: %d, time = %ds, training loss: %f"%(j+self.start_epoch, k, time.time() - begin, loss_val))
            
            if (j+self.start_epoch+1)%20 == 0:
                torch.save(self.D.state_dict(), os.path.join('results', self.args.exp_name, 'models', 'epoch_%d'%(j+self.start_epoch)))
                if self.args.save_train: # save the last batch results
                    output = (disparity_map.cpu().data.numpy())
                    data_util.save_height(self.args.exp_name, output, [fn+'_%d'%j for fn in filenames], 'disparity_train')
            print("Epochs %d, time = %ds, average training loss: %f"%(j+self.start_epoch, time.time() - begin, np.mean(train_epoch_loss)))
        
        # test
        for k, batch in enumerate(self.test_loader):
            img_pair, pre_disp,filenames = batch['images'].float(), batch['disps'].float(), batch['names']
            X = Variable(torch.cuda.FloatTensor(img_pair), requires_grad=False)
            Disp = Variable(torch.cuda.FloatTensor(np.expand_dims(np.expand_dims(pre_disp, 0), 0)), requires_grad=False)
            disparity_map = self.D(X)
            loss = self.L(Disp, disparity_map)
            loss_val = loss.data.cpu().numpy()
            test_epoch_loss.append(loss_val)
            output = (disparity_map.cpu().data.numpy())
            data_util.save_height(self.args.exp_name, output, filenames, 'disparity_test')
            del X,Y,disparity_map,loss
        print('overall performance: %f'%np.mean(test_epoch_loss))
        return test_epoch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, default='pretrain', help='the name to identify current experiment')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='How many epochs to run in total?')
    parser.add_argument("-ie", "--input_epoch", type=int, default=0, help='Load model after n epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=2020, help='random seed for generating data')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-m', '--model', type=str, default='plain', help='which model to be used')
    parser.add_argument('--img_size', type=int, default=900, help='number of folds')
    parser.add_argument('-st', '--save_train', action='store_true', help='save the results for training data')
    parser.add_argument("-ld", "--load_model", action='store_true', help='Load pretrained model or not')

    args = parser.parse_args()

    if not os.path.exists(os.path.join('results/', args.exp_name, 'models')):
        os.mkdir(os.path.join('results/', args.exp_name))
        os.mkdir(os.path.join('results/', args.exp_name, 'models'))
        os.mkdir(os.path.join('results/', args.exp_name, 'disparity_train'))
        os.mkdir(os.path.join('results/', args.exp_name, 'disparity_test'))

    # os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    filenames = [line.rstrip() for line in open('data/file_lists.txt')]

    ### load data from scratch. this requires a lot of geo packages###
    # data_path = '/disk/songwei/LockheedMartion/end2end/MVS/'
    # kml_path = '/disk/songwei/LockheedMartion/end2end/KML/'
    # gt_path = '/disk/songwei/LockheedMartion/end2end/DSM/'
    # train_dataset= data_util.MVSdataset(gt_path, data_path, kml_path, args.img_size, filenames)
    # train_dataset.save_data('results/data_small.npz')

    ### load data from numpy file###
    # data_file = 'data/data_small.npz'
    # train_loader, test_loader = data_util.get_numpy_dataset(filenames[-100:-20], args.batch_size, data_file)
    # data_file = 'data/data_all.npz'
    # train_loader, _, test_loader = data_util.get_numpy_dataset(filenames, args.batch_size, data_file, validation=False)
    data_file = 'data/data_masked.npz'
    train_loader, _, test_loader = data_util.get_masked_numpy_dataset(filenames, args.batch_size, data_file, validation=False)


    trainer = Trainer(args, train_loader, test_loader)
    if args.load_model:
        print('loading pretrained model from epoch %d'%args.input_epoch)
        # trainer.D.load_state_dict(torch.load(os.path.join('./results/pretrain_psmnet/models', 'epoch_%d'%args.input_epoch)))
        dic = torch.load(os.path.join('./results/pretrain_psmnet2/models', 'epoch_%d'%args.input_epoch))
        # state_dic_new = {key.replace('module.', ''): item for key, item in dic.items()}
        # import ipdb;ipdb.set_trace()
        trainer.D.load_state_dict(dic)
    trainer.run()
