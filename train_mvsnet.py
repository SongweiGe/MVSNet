import os
import time
import pyproj
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.data_util as data_util
from utils.model_util import FlowNetS

from utils import geo_utils
from triangulationRPC_matrix_torch import triangulationRPC_matrix
# from triangulation_bak import triangulationRPC_matrix
# from torchinterp1d import Interp1d

class Trainer(object):
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.n_epochs = args.epochs
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.start_epoch = args.input_epoch
        self.D = FlowNetS().cuda()
        # self.L = nn.MSELoss().cuda()
        self.L = nn.L1Loss().cuda()
        self.optimizer = torch.optim.Adadelta(self.D.parameters(), lr=1e-0)
        self.train_loader = train_loader
        self.test_loader = test_loader
        np.random.seed(2019)
        self.wgs84 = pyproj.Proj('+proj=utm +zone=21 +datum=WGS84 +south')
        # self.interp_method = Interp1d()
        self.cu1_rec, self.ru1_rec = torch.meshgrid([torch.arange(self.img_size), torch.arange(self.img_size)])
        self.cu1_rec = self.cu1_rec.type(torch.cuda.DoubleTensor).transpose(1, 0)
        self.ru1_rec = self.ru1_rec.type(torch.cuda.DoubleTensor).transpose(1, 0)
        self.xx, self.yy = np.meshgrid(np.arange(250), np.arange(250))

    def apply_homography(self, h, x):
        #                    h[0] h[1] h[2]
        # The convention is: h[3] h[4] h[5]
        #                    h[6] h[7] h[8]
        y0 = torch.zeros_like(x[0])
        y1 = torch.zeros_like(x[1])
        # tmp = x[0]
        z = h[6]*x[0] + h[7]*x[1] + h[8]
        y0 = (h[0]*x[0] + h[1]*x[1] + h[2]) / z
        y1 = (h[3]*x[0] + h[4]*x[1] + h[5]) / z
        return y0, y1

    def triangulation_forward(self, disparity_map, masks, rpc_pair, h_pair, area_info):
        masks = masks[0]
        rpc_l, rpc_r = rpc_pair[0]
        h_left_inv, h_right_inv = h_pair[0]

        # transform into numpy to process the X and Y coordinates
        area_info = area_info.data.numpy()
        bbox, bounds, im_size = area_info[0, :4], np.stack([area_info[0, 4:6], area_info[0, 6:8]]), area_info[0, -2:]
        cu2_rec = self.cu1_rec + disparity_map[0, :, :].type(torch.cuda.DoubleTensor)
        ru2_rec = self.ru1_rec
        # cu2_rec = cu1_rec

        cu1, ru1 = self.apply_homography(h_left_inv, [self.cu1_rec, self.ru1_rec])
        cu2, ru2 = self.apply_homography(h_right_inv, [cu2_rec, ru2_rec])

        ru1 = ru1[masks].reshape(-1)
        cu1 = cu1[masks].reshape(-1)
        ru2 = ru2[masks].reshape(-1)
        cu2 = cu2[masks].reshape(-1)
        Xu, Yu, Zu, _, _ = triangulationRPC_matrix(ru1[:630000], cu1[:630000], ru2[:630000], cu2[:630000], rpc_l, rpc_r, verbose=False, inverse_bs=1000)

        lons, lats = self.wgs84(Xu.cpu().data.numpy(), Yu.cpu().data.numpy())
        ix, iy = geo_utils.spherical_to_image_positions(lons, lats, bounds, im_size)
        # import ipdb;ipdb.set_trace()

        valid_points = np.logical_and(np.logical_and(iy>bbox[0], iy<bbox[0]+250), np.logical_and(ix>bbox[2], ix<bbox[2]+250))
        # input_coords = torch.stack([torch.cuda.FloatTensor(iy), torch.cuda.FloatTensor(ix)])
        # output_coords = torch.stack([torch.cuda.FloatTensor(self.yy), torch.cuda.FloatTensor(self.xx)])
        # int_im = self.interp_method(input_coords, Zu[valid_points], output_coords)
        # int_im = griddata((iy, ix), Zu[valid_points], (self.yy, self.xx))
        # int_im = geo_utils.fill_holes(int_im)
        # return int_im[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        return ix[valid_points]-bbox[2], iy[valid_points]-bbox[0], Zu[np.where(valid_points)], Xu, Yu, Zu

    def calculate_loss(self, X, Y, Z, gt):
        # import ipdb;ipdb.set_trace()
        x_max, y_max = gt.shape[1:]
        grid = torch.stack([torch.cuda.FloatTensor(X)/x_max*2-1, torch.cuda.FloatTensor(Y)/y_max*2-1]).transpose(1, 0).view(1, 1, -1, 2)
        gt_height = torch.nn.functional.grid_sample(gt.unsqueeze(1), grid.cuda()).squeeze()
        return self.L(gt_height.double(), Z)

    def run(self):
        self.D.train()
        loss_val = float('inf')
        self.train_loss = []
        outputs = []
        print('start training')

        for j in range(self.n_epochs):
            begin = time.time()
            train_epoch_loss = []
            test_epoch_loss = []
            for k, batch in enumerate(self.train_loader):
                #forward calculation and back propagation, X: B x P x 2 x W x H
                X, masks, h_pair, rpc_pair, area_info, Y, filenames = batch['images'].cuda(), batch['masks'], batch['hs'].cuda(),\
                                                                    batch['rpcs'].cuda(), batch['area_infos'], batch['ys'].cuda(), batch['names']
                
                self.optimizer.zero_grad()
                disparity_map = self.D(X)[0][:, 0, :, :] # only left disp map
                lon, lat, heights, Xu, Yu, Zu = self.triangulation_forward(disparity_map, masks, rpc_pair, h_pair, area_info)
                # import ipdb;ipdb.set_trace()
                # print('range of predicted height: (%.3f, %.3f), ground truth: (%.3f, %.3f)'%(heights.min(), heights.max(), Y.min(), Y.max()))
                loss = self.calculate_loss(lon, lat, heights, Y[0:1])
                loss_val = loss.data.cpu().numpy()
                # print('The number of remaining points:%d'%len(lon))
                if np.isnan(loss_val):
                    continue
                    # import ipdb;ipdb.set_trace()
                loss.backward()
                self.optimizer.step()
                train_epoch_loss.append(loss_val)
                del X,Y,lon, lat, heights,loss
                print("Epochs %d, iteration: %d, time = %ds, training loss: %f"%(j+self.start_epoch, k, time.time() - begin, loss_val))
            
            if (j+self.start_epoch+1)%5 == 0:
                torch.save(self.D.state_dict(), os.path.join('results', self.args.exp_name, 'models', 'epoch_%d'%(j+self.start_epoch)))
            print("Epochs %d, time = %ds, average training loss: %f"%(j+self.start_epoch, time.time() - begin, np.mean(train_epoch_loss)))
        
        # save the last training estimation
        if self.args.save_train:
            output = (pred_height.cpu().data.numpy())
            data_util.save_height(self.args.exp_name, output, filenames[train_batch_ids], 'train')
        # test
        for k, batch in enumerate(self.test_loader):
            X, masks, h_pair, rpc_pair, area_info, Y, filenames = batch['images'].cuda(), batch['masks'], batch['hs'].cuda(),\
                                                                batch['rpcs'].cuda(), batch['area_infos'], batch['ys'].cuda(), batch['names']
            lon, lat, heights, Xu, Yu, Zu = self.triangulation_forward(disparity_map, masks, rpc_pair, h_pair, area_info)
            loss = self.calculate_loss(lon, lat, heights, Y[0:1])
            loss_val = loss.data.cpu().numpy()
            test_epoch_loss.append(loss_val)
            output = (pred_height.cpu().data.numpy())
            data_util.save_height(self.args.exp_name, output, filenames[test_batch_ids], 'test')
            del X,Y,pred_height,loss
        print('overall performance: %f'%np.mean(test_epoch_loss))
        return test_epoch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, default='rpcnet', help='the name to identify current experiment')
    parser.add_argument("-ie", "--input_epoch", type=int, default=0, help='Load model after n epochs')
    parser.add_argument("-ld", "--load_model", type=int, default=0, help='Load pretrained model or not')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-m', '--model', type=str, default='plain', help='which model to be used')
    parser.add_argument('-r', '--res', type=int, default=0, help='residual or not')
    parser.add_argument('--img_size', type=int, default=1200, help='number of folds')
    parser.add_argument('--save_train', type=bool, default=False, help='save the reconstruction results for training data')

    args = parser.parse_args()

    if not os.path.exists(os.path.join('results/', args.exp_name, 'models')):
        os.mkdir(os.path.join('results/', args.exp_name))
        os.mkdir(os.path.join('results/', args.exp_name, 'models'))
        os.mkdir(os.path.join('results/', args.exp_name, 'reconstruction_train'))
        os.mkdir(os.path.join('results/', args.exp_name, 'reconstruction_test'))

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
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
    data_file = 'data/data_all.npz'
    train_loader, test_loader = data_util.get_numpy_dataset(filenames, args.batch_size, data_file)

    trainer = Trainer(args, train_loader, test_loader)
    if args.load_model:
        print('loading pretrained model from epoch %d'%args.input_epoch)
        trainer.D.load_state_dict(torch.load(os.path.join('./results/pretrain/models', 'epoch_%d'%args.input_epoch)))
    trainer.run()
