import os
import time
import pyproj
import argparse
import numpy as np
import imageio
import scipy.misc
from scipy.interpolate import griddata
import imageio as misc

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import eval_util, data_util, geo_utils
from triangulationRPC_matrix_torch import triangulationRPC_matrix

result_path = './results/baseline_triangulation'

if not os.path.exists(result_path):
    os.mkdir(result_path)

fire_palette = misc.imread(os.path.join('image/fire_palette.png'))[0][:, 0:3]

wgs84 = pyproj.Proj('+proj=utm +zone=21 +datum=WGS84 +south')

def apply_homography(h, x):
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


def triangulation_forward(disparity_map, masks, nrow, ncol, rpc_pair, h_pair, area_info):
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

    cu1, ru1 = apply_homography(h_left_inv, [cu1_rec, ru1_rec])
    cu2, ru2 = apply_homography(h_right_inv, [cu2_rec, ru2_rec])

    ru1 = ru1[masks].reshape(-1)
    cu1 = cu1[masks].reshape(-1)
    ru2 = ru2[masks].reshape(-1)
    cu2 = cu2[masks].reshape(-1)
    Xu, Yu, Zu, _, _ = triangulationRPC_matrix(ru1[:630000], cu1[:630000], ru2[:630000], cu2[:630000], rpc_l, rpc_r, verbose=False, inverse_bs=1000)

    xx, yy = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
    lons, lats = wgs84(Xu.cpu().data.numpy(), Yu.cpu().data.numpy())
    ix, iy = geo_utils.spherical_to_image_positions(lons, lats, bounds, im_size)
    valid_points = np.logical_and(np.logical_and(iy>bbox[0], iy<bbox[0]+250), np.logical_and(ix>bbox[2], ix<bbox[2]+250))
    int_im = griddata((iy, ix), Zu.cpu().data.numpy(), (yy, xx))
    int_im = geo_utils.fill_holes(int_im)
    # import ipdb;ipdb.set_trace()
    return int_im[int(bbox[0]):int(bbox[0]+250), int(bbox[2]):int(bbox[2]+250)], int_im, ix-bbox[2], iy-bbox[0], Zu


def run_baseline(train_loader, test_loader):
    # rsmes = []
    # accs = []
    # coms = []
    # l1es = []
    # for k, batch in enumerate(train_loader):
    #     X, Disp, masks, h_pair, rpc_pair, area_info, Y, filenames = batch['images'].cuda(), batch['disps'].cuda(), batch['masks'], batch['hs'].cuda(),\
    #                                                         batch['rpcs'].cuda(), batch['area_infos'], batch['ys'].cuda(), batch['names']
    #     height_map, height_map_all, lon, lat, heights = triangulation_forward(Disp, masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
    #     # import ipdb;ipdb.set_trace()
        
    #     rsme, acc, com, l1e = eval_util.evaluate(height_map, Y[0].cpu())
    #     rsmes.append(rsme)
    #     accs.append(acc)
    #     coms.append(com)
    #     l1es.append(l1e)

    # print('The average training errors are: rsme = %.4f +/- %.4f, acc = %.4f +/- %.4f, com = %.4f +/- %.4f, l1e = %.4f +/- %.4f\n'%(
    #     np.mean(rsmes), np.std(rsmes), np.mean(accs), np.std(accs), np.mean(coms), np.std(coms), np.mean(l1es), np.std(l1es)))
    
    # test
    rsmes = []
    accs = []
    coms = []
    l1es = []
    for k, batch in enumerate(test_loader):
        X, Disp, masks, h_pair, rpc_pair, area_info, Y, filenames = batch['images'].cuda(), batch['disps'].cuda(), batch['masks'], batch['hs'].cuda(),\
                                                            batch['rpcs'].cuda(), batch['area_infos'], batch['ys'].cuda(), batch['names']
        out_path = os.path.join(result_path, filenames[0])
        height_map, height_map_all, lon, lat, heights = triangulation_forward(Disp, masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
        rsme, acc, com, l1e = eval_util.evaluate(height_map, Y[0].cpu())
        color_map = eval_util.getColorMapFromPalette(height_map, fire_palette)
        misc.imsave(out_path+'_gt_data.png', color_map)
        rsmes.append(rsme)
        accs.append(acc)
        coms.append(com)
        l1es.append(l1e)

    print('The average test errors are: rsme = %.4f +/- %.4f, acc = %.4f +/- %.4f, com = %.4f +/- %.4f, l1e = %.4f +/- %.4f\n'%(
        np.mean(rsmes), np.std(rsmes), np.mean(accs), np.std(accs), np.mean(coms), np.std(coms), np.mean(l1es), np.std(l1es)))




if __name__ == '__main__':    
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    filenames = [line.rstrip() for line in open('data/file_lists.txt')]

    ### load data from numpy file###
    data_file = 'data/data_all.npz'
    train_loader, _, test_loader = data_util.get_numpy_dataset(filenames, 1, data_file, validation=False)

    # data_file = './data/data_small.npz'
    # train_loader, _, test_loader = data_util.get_numpy_dataset(filenames[-100:-20], 1, data_file, validation=False)
    run_baseline(train_loader, test_loader)
