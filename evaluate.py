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
        masks = masks[0]
        rpc_l, rpc_r = rpc_pair[0]
        h_left_inv, h_right_inv = h_pair[0]

        # transform into numpy to process the X and Y coordinates
        cu1_rec, ru1_rec = torch.meshgrid([torch.arange(ncol), torch.arange(nrow)])
        cu1_rec = cu1_rec.type(torch.cuda.DoubleTensor).transpose(1, 0)
        ru1_rec = ru1_rec.type(torch.cuda.DoubleTensor).transpose(1, 0)
        area_info = area_info.data.numpy()
        bbox, bounds, im_size = area_info[0, :4], np.stack([area_info[0, 4:6], area_info[0, 6:8]]), area_info[0, -2:]
        cu2_rec = cu1_rec + disparity_map[0, :, :].type(torch.cuda.DoubleTensor)
        ru2_rec = ru1_rec
        # cu2_rec = cu1_rec

        cu1, ru1 = self.apply_homography(h_left_inv, [cu1_rec, ru1_rec])
        cu2, ru2 = self.apply_homography(h_right_inv, [cu2_rec, ru2_rec])

        ru1 = ru1[masks].reshape(-1)
        cu1 = cu1[masks].reshape(-1)
        ru2 = ru2[masks].reshape(-1)
        cu2 = cu2[masks].reshape(-1)
        Xu, Yu, Zu, _, _ = triangulationRPC_matrix(ru1[:630000], cu1[:630000], ru2[:630000], cu2[:630000], rpc_l, rpc_r, verbose=False, inverse_bs=1000)

        xx, yy = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
        lons, lats = self.wgs84(Xu.cpu().data.numpy(), Yu.cpu().data.numpy())
        ix, iy = geo_utils.spherical_to_image_positions(lons, lats, bounds, im_size)
        valid_points = np.logical_and(np.logical_and(iy>bbox[0], iy<bbox[0]+250), np.logical_and(ix>bbox[2], ix<bbox[2]+250))
        int_im = griddata((iy, ix), Zu.cpu().data.numpy(), (yy, xx))
        int_im = geo_utils.fill_holes(int_im)
        # import ipdb;ipdb.set_trace()
        return int_im[int(bbox[0]):int(bbox[0]+250), int(bbox[2]):int(bbox[2]+250)], int_im, ix-bbox[2], iy-bbox[0], Zu

    def calculate_loss(self, X, Y, Z, gt):
        x_max, y_max = gt.shape[1:]
        grid = torch.stack([torch.cuda.FloatTensor(X)/x_max, torch.cuda.FloatTensor(Y)/y_max]).transpose(1, 0).view(1, 1, -1, 2)
        gt_height = torch.nn.functional.grid_sample(gt.unsqueeze(1), grid.cuda()).squeeze()
        return self.L(gt_height.double(), Z.cuda())

    def forward(self, sample):
        X, masks, h_pair, rpc_pair, area_info, Y, filenames = batch['images'].cuda(), batch['masks'], batch['hs'].cuda(),\
                                                            batch['rpcs'].cuda(), batch['area_infos'], batch['ys'].cuda(), batch['names']
        disparity_map = self.D(X)[0][:, 0, :, :]
        # import ipdb;ipdb.set_trace()
        # lon, lat, heights, Xu, Yu, Zu = self.triangulation_forward(disparity_map, masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
        height_map, height_map_all, lon, lat, heights = self.triangulation_forward(disparity_map, masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
        # _, _, lon_zero, lat_zero, heights_zero = self.triangulation_forward(torch.zeros_like(disparity_map), masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
        # np.sqrt(np.sum((lon-lon_zero)**2))
        # np.sqrt(np.sum((lat-lat_zero)**2))
        # np.sqrt(np.sum((heights-heights_zero)**2))
        # imageio.imsave(os.path.join('debug/', 'rand_disp.png'), Disp[0, 0].cpu().data.numpy())
        # imageio.imsave(os.path.join('debug/', 'good_disp.png'), disparity_map[0, 0].cpu().data.numpy())
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
        disparity_map = self.D(X)[0]+Disp
        # disparity_map = Disp
        height_map, height_map_all, lon, lat, heights = self.triangulation_forward(disparity_map, masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
        loss = self.calculate_loss(lon, lat, heights, Y)
        return disparity_map, height_map, height_map_all, loss
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, default='plain', help='the name to identify current experiment')
    parser.add_argument("-ie", "--input_epoch", type=int, default=28, help='Load model after n epochs')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-nf', '--n_folds', type=int, default=5, help='number of folds')
    parser.add_argument('-r', '--res', type=int, default=0, help='residual or not')
    parser.add_argument('-g', '--gpu_id', type=str, default='1', help='gpuid used for trianing')
    parser.add_argument('-m', '--model', type=str, default='plain', help='which model to be used')
    parser.add_argument('--save_train', type=bool, default=False, help='save the reconstruction results for training data')

    args = parser.parse_args()

    debug_path = './debug/end2end_%s_%d'%(args.exp_name, args.input_epoch)
    debug_path_train = os.path.join(debug_path, 'train')
    debug_path_test = os.path.join(debug_path, 'test')
    if not os.path.exists(debug_path):
        os.mkdir(debug_path)
        os.mkdir(debug_path_train)
        os.mkdir(debug_path_test)
    fire_palette = imageio.imread('./image/fire_palette.png')[0][:, 0:3]
    fw = open(os.path.join(debug_path, 'log%d.txt'%args.input_epoch), 'w')

    rsmes = []
    accs = []
    coms = []
    l1es = []

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    filenames = [line.rstrip() for line in open('data/file_lists.txt')]
    # test_dataset= data_util.MVSdataset_lithium(data_file)

    data_file = './data/data_small.npz'
    train_loader, test_loader = data_util.get_numpy_dataset(filenames[-100:-20], args.batch_size, data_file)

    # data_file = 'data/data_all.npz'
    # train_loader, test_loader = data_util.get_numpy_dataset(filenames, args.batch_size, data_file)
    # filenames = [line.split('/')[6] for line in open('debug/log.txt') if line.startswith('/disk')]
    # test_dataset= data_util.MVSdataset(gt_path, data_path, kml_path, filenames)
    Predictor = Predictor(args)
    Predictor.D.load_state_dict(torch.load(os.path.join('./results', args.exp_name, 'models', 'epoch_%d'%(args.input_epoch))))

    for i, batch in enumerate(train_loader):
        dmap, hmap, hmap_all, loss = Predictor.forward(batch)
        left_img = batch['images'][0]
        gt_data = batch['ys'][0]
        rsme, acc, com, l1e = eval_util.evaluate(hmap, gt_data)
        rsmes.append(rsme)
        accs.append(acc)
        coms.append(com)
        l1es.append(l1e)
        fw.write('The errors for asp are: loss = %.4f, rsme = %.4f, acc = %.4f, com = %.4f, l1e = %.4f\n'%(loss.cpu().data, rsme, acc, com, l1e))

        # left_img = geo_utils.open_gtiff(os.path.join(data_path, filenames[i], 'cropped_images', '02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF.tif'))
        # imageio.imsave(os.path.join(debug_path, filenames[i]+'_leftimg.png'), left_img[bbox[0]:bbox[1], bbox[2]:bbox[3]])

        # import ipdb;ipdb.set_trace()
        imageio.imsave(os.path.join(debug_path_train, batch['names'][0]+'_disp.png'), dmap[0].cpu().data.numpy())
        color_map = eval_util.getColorMapFromPalette(dmap[0].cpu().data.numpy(), fire_palette)
        imageio.imsave(os.path.join(debug_path_train, batch['names'][0]+'_disp_color.png'), color_map)
        imageio.imsave(os.path.join(debug_path_train, batch['names'][0]+'_left.png'), left_img[0])
        color_map = eval_util.getColorMapFromPalette(hmap, fire_palette)
        imageio.imsave(os.path.join(debug_path_train, batch['names'][0]+'_height.png'), color_map)
        color_map = eval_util.getColorMapFromPalette(hmap_all, fire_palette)
        imageio.imsave(os.path.join(debug_path_train, batch['names'][0]+'_height_all.png'), color_map)
        color_map = eval_util.getColorMapFromPalette(gt_data.data.numpy(), fire_palette)
        imageio.imsave(os.path.join(debug_path_train, batch['names'][0]+'_gt.png'), color_map)

    for i, batch in enumerate(test_loader):
        dmap, hmap, hmap_all, loss = Predictor.forward(batch)
        left_img = batch['images'][0]
        gt_data = batch['ys'][0]
        rsme, acc, com, l1e = eval_util.evaluate(hmap, gt_data)
        rsmes.append(rsme)
        accs.append(acc)
        coms.append(com)
        l1es.append(l1e)
        fw.write('The errors for asp are: loss = %.4f, rsme = %.4f, acc = %.4f, com = %.4f, l1e = %.4f\n'%(loss.cpu().data, rsme, acc, com, l1e))

        # left_img = geo_utils.open_gtiff(os.path.join(data_path, filenames[i], 'cropped_images', '02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF.tif'))
        # imageio.imsave(os.path.join(debug_path, filenames[i]+'_leftimg.png'), left_img[bbox[0]:bbox[1], bbox[2]:bbox[3]])

        # import ipdb;ipdb.set_trace()
        imageio.imsave(os.path.join(debug_path_test, batch['names'][0]+'_disp.png'), dmap[0].cpu().data.numpy())
        color_map = eval_util.getColorMapFromPalette(dmap[0].cpu().data.numpy(), fire_palette)
        imageio.imsave(os.path.join(debug_path_test, batch['names'][0]+'_disp_color.png'), color_map)
        imageio.imsave(os.path.join(debug_path_test, batch['names'][0]+'_left.png'), left_img[0])
        color_map = eval_util.getColorMapFromPalette(hmap, fire_palette)
        imageio.imsave(os.path.join(debug_path_test, batch['names'][0]+'_height.png'), color_map)
        color_map = eval_util.getColorMapFromPalette(hmap_all, fire_palette)
        imageio.imsave(os.path.join(debug_path_test, batch['names'][0]+'_height_all.png'), color_map)
        color_map = eval_util.getColorMapFromPalette(gt_data.data.numpy(), fire_palette)
        imageio.imsave(os.path.join(debug_path_test, batch['names'][0]+'_gt.png'), color_map)

        del dmap, hmap, hmap_all, loss

    fw.write('The average errors for asp are: rsme = %.4f +/- %.4f, acc = %.4f +/- %.4f, com = %.4f +/- %.4f, l1e = %.4f +/- %.4f\n'%(np.mean(rsmes), 
        np.std(rsmes), np.mean(accs), np.std(accs), np.mean(coms), np.std(coms), np.mean(l1es), np.std(l1es)))
