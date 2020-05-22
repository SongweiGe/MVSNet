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
from utils.basic import PSMNet

from utils import geo_utils, eval_util
from triangulationRPC_matrix_torch import triangulationRPC_matrix

class Predictor(object):
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.batch_size = args.batch_size
        # self.D = FlowNetS().cuda()
        # model = nn.DataParallel(PSMNet(maxdisp=48))
        model = PSMNet(maxdisp=48)
        self.D = model.cuda()
        self.L = nn.L1Loss().cuda()
        self.train_loader = train_loader
        self.test_loader = test_loader
        np.random.seed(args.seed)

    def run(self, save_fig=False):
        # net_init(self.D)
        self.D.train()
        train_losses = []
        test_losses = []
        outputs = []
        # print('start evaluation')
        confidence_palette = imageio.imread('./image/confidence_palette.png')[0][:, 0:3]

        for k, batch in enumerate(self.train_loader):
            img_pair, Disp, filenames = batch['images'].cuda(), batch['disps'].cuda(), batch['names']
            # disparity_map = self.D(img_pair.cuda())[0][:, 0, :, :]
            import ipdb;ipdb.set_trace()
            disparity_map = self.D(img_pair[:, 0:1, :args.img_size, :args.img_size], img_pair[:, 1:2, :args.img_size, :args.img_size]) # PSMNet
            loss = self.L(Disp[:, :args.img_size, :args.img_size], disparity_map)

            loss_val = loss.data.cpu().numpy()
            train_losses.append(loss_val)
            if save_fig:
                for dmap, dmap_gt, fn in zip(disparity_map, Disp, filenames):
                    imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_train', fn+'_disp.png'), eval_util.getColorMapFromPalette(dmap.cpu().data.numpy(), confidence_palette, im_min=-5, im_max=5))
                    imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_train', fn+'_disp_gt.png'), eval_util.getColorMapFromPalette(dmap_gt.cpu().data.numpy()[:, :args.img_size, :args.img_size], confidence_palette, im_min=-5, im_max=5))
            del img_pair,Disp,loss,batch
            
        # test
        for k, batch in enumerate(self.test_loader):
            img_pair, Disp, filenames = batch['images'].cuda(), batch['disps'].cuda(), batch['names']
            # disparity_map = self.D(img_pair.cuda())[0][:, 0, :args.img_size, :args.img_size]
            disparity_map = self.D(img_pair[:, 0:1, :args.img_size, :args.img_size], img_pair[:, 1:2, :args.img_size, :args.img_size]) # PSMNet
            loss = self.L(Disp[:, :args.img_size, :args.img_size], disparity_map)
            loss_val = loss.data.cpu().numpy()
            test_losses.append(loss_val)

            if save_fig:
                for dmap, dmap_gt, fn in zip(disparity_map, Disp, filenames):
                    imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_test', fn+'_disp.png'), eval_util.getColorMapFromPalette(dmap.cpu().data.numpy(), confidence_palette, im_min=-5, im_max=5))
                    imageio.imsave(os.path.join('results', self.args.exp_name, 'disparity_test', fn+'_disp_gt.png'), eval_util.getColorMapFromPalette(dmap_gt.cpu().data.numpy()[:, :args.img_size, :args.img_size], confidence_palette, im_min=-5, im_max=5))
            del img_pair,Disp,loss
            
        print("Average training loss: %f, Average testing loss: %f"%(np.mean(train_losses), np.mean(test_losses)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, default='pretrain_psmnet3', help='the name to identify current experiment')
    parser.add_argument("-ie", "--input_epoch", type=int, default=399, help='Load model after n epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-g', '--gpu_id', type=str, default='1', help='gpuid used for trianing')
    parser.add_argument('--img_size', type=int, default=900, help='number of folds')
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
    # data_file = './data/data_small.npz'
    # train_loader, valid_loader, test_loader = data_util.get_numpy_dataset(filenames[-100:-20], args.batch_size, data_file, validation=False)

    data_file = 'data/data_all.npz'
    train_loader, valid_loader, test_loader = data_util.get_numpy_dataset(filenames, args.batch_size, data_file, validation=False)
    
    # filenames = [line.split('/')[6] for line in open('debug/log.txt') if line.startswith('/disk')]
    # test_dataset= data_util.MVSdataset(gt_path, data_path, kml_path, filenames)

    # predictor.D.load_state_dict(torch.load(os.path.join('./results', args.exp_name, 'models', 'epoch_%d'%(args.input_epoch))))
    for epoch in range(399, 419, 20):
        print('epoch: %d'%epoch)
        # import ipdb;ipdb.set_trace()
        predictor = Predictor(args, train_loader, test_loader)
        # predictor.D.load_state_dict(torch.load(os.path.join('./results', args.exp_name, 'models', 'epoch_%d'%epoch)))
        dic = torch.load(os.path.join('./results', args.exp_name, 'models', 'epoch_%d'%epoch))
        state_dic_new = {key.replace('module.', ''): item for key, item in dic.items()}
        predictor.D.load_state_dict(state_dic_new)
        save_fig = True if epoch == 399 else False
        predictor.run(save_fig)
 