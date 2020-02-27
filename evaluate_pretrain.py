import os
import time
import pyproj
import imageio
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.data_util as data_util
from utils.model_util import FlowNetS, net_init

from utils import geo_utils

class Predictor(object):
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.batch_size = args.batch_size
        self.D = FlowNetS().cuda()
        self.L = nn.L1Loss().cuda()
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
        print('start evaluation')

        # for k, batch in enumerate(self.train_loader):
        #     X, Disp, filenames = batch['images'].cuda(), batch['disps'].cuda(), batch['names']
        #     disparity_map = self.D(X)[0][:, 0, :, :]
        #     # import ipdb;ipdb.set_trace()
        #     for x, dmap, dmap_gt, fn in zip(X, disparity_map, Disp, filenames):
        #         imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_train', fn+'_input.png'), x[0].cpu().data.numpy())
        #         imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_train', fn+'_disp.png'), dmap.cpu().data.numpy())
        #         imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_train', fn+'_disp_gt.png'), dmap_gt.cpu().data.numpy())

        # test
        for k, batch in enumerate(self.test_loader):
            X, Disp,filenames = batch['images'].cuda(), batch['disps'].cuda(), batch['names']
            disparity_map = self.D(X)[0][:, 0, :, :]
            for x, dmap, dmap_gt, fn in zip(X, disparity_map, Disp, filenames):
                imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_test', fn+'_input.png'), x[0].cpu().data.numpy())
                imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_test', fn+'_disp.png'), dmap.cpu().data.numpy())
                imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_test', fn+'_disp_gt.png'), dmap_gt.cpu().data.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, default='pretrain', help='the name to identify current experiment')
    parser.add_argument("-ie", "--input_epoch", type=int, default=399, help='Load model after n epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-g', '--gpu_id', type=str, default='1', help='gpuid used for trianing')
    parser.add_argument('--save_train', type=bool, default=False, help='save the reconstruction results for training data')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    filenames = [line.rstrip() for line in open('data/file_lists.txt')]

    ### load data from scratch. this requires a lot of geo packages###
    # data_path = '/disk/songwei/LockheedMartion/end2end/MVS/'
    # kml_path = '/disk/songwei/LockheedMartion/end2end/KML/'
    # gt_path = '/disk/songwei/LockheedMartion/end2end/DSM/'
    # train_dataset= data_util.MVSdataset(gt_path, data_path, kml_path, args.img_size, filenames)
    # train_dataset.save_data('results/data_small.npz')

    ### load data from numpy file###
    # data_file = 'data/data_all.npz'
    # train_loader, test_loader = data_util.get_numpy_dataset(filenames, args.batch_size, data_file)

    data_file = './data/data_small.npz'
    train_loader, test_loader = data_util.get_numpy_dataset(filenames[-100:-20], args.batch_size, data_file)

    predictor = Predictor(args, train_loader, test_loader)
    predictor.D.load_state_dict(torch.load(os.path.join('./results', args.exp_name, 'models', 'epoch_%d'%(args.input_epoch))))
    predictor.run()
