import os
import time
import pyproj
import argparse
import numpy as np
import imageio
import scipy.misc
from scipy.interpolate import griddata

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.data_util as data_util
from utils.model_util import FlowNetS

from utils import geo_utils, eval_util
from triangulationRPC_matrix_torch import triangulationRPC_matrix
# from torchinterp1d import Interp1d

class Predictor(object):
    def __init__(self, args):
        self.D = FlowNetS().cuda()
        self.wgs84 = pyproj.Proj('+proj=utm +zone=21 +datum=WGS84 +south')
        self.L = nn.L1Loss().cuda()

    def weights_init(self, m):
        try:
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        except:
            m.data.normal_(0, 0.01)

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

    def triangulation_forward(self, disparity_map, masks, nrow, ncol, rpc_pair, h_pair, area_info):
        rpc_l, rpc_r = rpc_pair
        h_left_inv, h_right_inv = h_pair
        bbox, bounds, im_size = area_info
        cu1_rec, ru1_rec = torch.meshgrid([torch.arange(ncol), torch.arange(nrow)])
        cu1_rec = cu1_rec.type(torch.cuda.DoubleTensor).transpose(1, 0)
        ru1_rec = ru1_rec.type(torch.cuda.DoubleTensor).transpose(1, 0)
        cu2_rec = cu1_rec + disparity_map[0, 0, :, :].type(torch.cuda.DoubleTensor)
        ru2_rec = ru1_rec
        # cu2_rec = cu1_rec

        cu1, ru1 = self.apply_homography(h_left_inv, [cu1_rec, ru1_rec])
        cu2, ru2 = self.apply_homography(h_right_inv, [cu2_rec, ru2_rec])
        # import ipdb;ipdb.set_trace()
        ru1 = ru1[masks].reshape(-1)
        cu1 = cu1[masks].reshape(-1)
        ru2 = ru2[masks].reshape(-1)
        cu2 = cu2[masks].reshape(-1)
        Xu, Yu, Zu, _, _ = triangulationRPC_matrix(ru1, cu1, ru2, cu2, rpc_l, rpc_r, verbose=False, inverse_bs=1000)

        xx, yy = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
        lons, lats = self.wgs84(Xu.cpu().data.numpy(), Yu.cpu().data.numpy())
        ix, iy = geo_utils.spherical_to_image_positions(lons, lats, bounds, im_size)
        valid_points = np.logical_and(np.logical_and(iy>bbox[0], iy<bbox[0]+250), np.logical_and(ix>bbox[2], ix<bbox[2]+250))
        int_im = griddata((iy, ix), Zu.cpu().data.numpy(), (yy, xx))
        int_im = geo_utils.fill_holes(int_im)
        return int_im[bbox[0]:bbox[0]+250, bbox[2]:bbox[2]+250], int_im, ix[valid_points]-bbox[2], iy[valid_points]-bbox[0], Zu[np.where(valid_points)]

    def calculate_loss(self, X, Y, Z, gt):
        x_max, y_max = gt.shape[1:]
        grid = torch.stack([torch.cuda.FloatTensor(X)/x_max, torch.cuda.FloatTensor(Y)/y_max]).transpose(1, 0).view(1, 1, -1, 2)
        gt_height = torch.nn.functional.grid_sample(gt.unsqueeze(1), grid.cuda()).squeeze()
        return self.L(gt_height, Z.cuda())

    def forward(self, sample):
        img_pair, masks, h_pair, rpc_pair, area_info, _, y = sample
        X = Variable(torch.cuda.FloatTensor([img_pair]), requires_grad=False)
        Y = Variable(torch.cuda.FloatTensor([y]), requires_grad=False)
        h_pair = torch.cuda.DoubleTensor(np.stack(h_pair))
        masks = torch.tensor(masks.tolist())
        disparity_map = self.D(X)[0]
        height_map, height_map_all, lon, lat, heights = self.triangulation_forward(disparity_map, masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
        loss = self.calculate_loss(lon, lat, heights, Y)
        return disparity_map, height_map, height_map_all, loss

    def forward_res(self, sample):
        img_pair, masks, h_pair, rpc_pair, area_info, pre_disp, y = sample
        X = Variable(torch.cuda.FloatTensor([np.vstack([img_pair, np.expand_dims(pre_disp, 0)])]), requires_grad=False)
        Y = Variable(torch.cuda.FloatTensor([y]), requires_grad=False)
        Disp = Variable(torch.cuda.FloatTensor(np.expand_dims(np.expand_dims(pre_disp, 0), 0)), requires_grad=False)
        h_pair = torch.cuda.DoubleTensor(np.stack(h_pair))
        masks = torch.tensor(masks.tolist())

        # disparity_map = self.D(X)[0]
        # disparity_map = self.D(X)[0]+Disp
        disparity_map = Disp
        height_map, height_map_all, lon, lat, heights = self.triangulation_forward(disparity_map, masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
        loss = self.calculate_loss(lon, lat, heights, Y)
        return disparity_map, height_map, height_map_all, loss
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, default='plain', help='the name to identify current experiment')
    parser.add_argument("-ie", "--input_epoch", type=int, default=28, help='Load model after n epochs')
    parser.add_argument("-ip", "--input_fold", type=str, default='0', help='Load model filepath')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-nf', '--n_folds', type=int, default=5, help='number of folds')
    parser.add_argument('-g', '--gpu_id', type=str, default='1', help='gpuid used for trianing')
    parser.add_argument('-m', '--model', type=str, default='plain', help='which model to be used')
    parser.add_argument('--save_train', type=bool, default=False, help='save the reconstruction results for training data')

    args = parser.parse_args()

    debug_path = './debug/end2end_%d'%args.input_epoch
    if not os.path.exists(debug_path):
        os.mkdir(debug_path)
    fire_palette = imageio.imread('./image/fire_palette.png')[0][:, 0:3]
    fw = open(os.path.join(debug_path, 'log%d.txt'%args.input_epoch), 'w')

    rsmes = []
    accs = []
    coms = []
    l1es = []

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    data_path = '/disk/songwei/LockheedMartion/end2end/MVS/'
    kml_path = '/disk/songwei/LockheedMartion/end2end/KML/'
    gt_path = '/disk/songwei/LockheedMartion/end2end/DSM/'
    data_file = './results/data_small.npz'
    test_dataset= data_util.MVSdataset_lithium(data_file)
    filenames = [line.split('/')[6] for line in open('debug/log.txt') if line.startswith('/disk')]
    # test_dataset= data_util.MVSdataset(gt_path, data_path, kml_path, filenames)
    Predictor = Predictor(args)
    # Predictor.D.load_state_dict(torch.load(os.path.join('./results', args.exp_name, 'models', 'fold%s_%d'%(args.input_fold, args.input_epoch))))

    # import ipdb;ipdb.set_trace()

    for i in range(len(test_dataset)):
        sample = test_dataset.__getitem__(i)
        bbox = sample[-3][0]
        dmap, hmap, hmap_all, loss = Predictor.forward_res(sample)
        left_img = sample[0][0]
        gt_data = sample[-1]
        rsme, acc, com, l1e = eval_util.evaluate(hmap, gt_data)
        rsmes.append(rsme)
        accs.append(acc)
        coms.append(com)
        l1es.append(l1e)
        fw.write('The errors for asp are: loss = %.4f, rsme = %.4f, acc = %.4f, com = %.4f, l1e = %.4f\n'%(loss.cpu().data, rsme, acc, com, l1e))

        # left_img = geo_utils.open_gtiff(os.path.join(data_path, filenames[i], 'cropped_images', '02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF.tif'))
        # imageio.imsave(os.path.join(debug_path, filenames[i]+'_leftimg.png'), left_img[bbox[0]:bbox[1], bbox[2]:bbox[3]])

        imageio.imsave(os.path.join(debug_path, filenames[i]+'_disp.png'), dmap[0, 0].cpu().data.numpy())
        color_map = eval_util.getColorMapFromPalette(dmap[0, 0].cpu().data.numpy(), fire_palette)
        imageio.imsave(os.path.join(debug_path, filenames[i]+'_disp_color.png'), color_map)
        imageio.imsave(os.path.join(debug_path, filenames[i]+'_left.png'), left_img)
        color_map = eval_util.getColorMapFromPalette(hmap, fire_palette)
        imageio.imsave(os.path.join(debug_path, filenames[i]+'_height.png'), color_map)
        color_map = eval_util.getColorMapFromPalette(hmap_all, fire_palette)
        imageio.imsave(os.path.join(debug_path, filenames[i]+'_height_all.png'), color_map)
        color_map = eval_util.getColorMapFromPalette(gt_data, fire_palette)
        imageio.imsave(os.path.join(debug_path, filenames[i]+'_gt.png'), color_map)

        del dmap, hmap, hmap_all, loss, sample

    fw.write('The average errors for asp are: rsme = %.4f +/- %.4f, acc = %.4f +/- %.4f, com = %.4f +/- %.4f, l1e = %.4f +/- %.4f\n'%(np.mean(rsmes), 
        np.std(rsmes), np.mean(accs), np.std(accs), np.mean(coms), np.std(coms), np.mean(l1es), np.std(l1es)))
