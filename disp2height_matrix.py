import os
import cv2
import time
import scipy.misc
import gdal
import OpenEXR
import numpy as np
import matplotlib.pyplot as plt
from rpcm.rpc_model import rpc_from_geotiff
from triangulationRPC_matrix import triangulationRPC_matrix
from scipy.interpolate import interp2d

import pyproj
from utils import geo_utils, eval_util
from scipy.interpolate import griddata

pairs_filenames = [
            ['02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF', 
            '02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.NTF'], 
]
pair_id = 0

def open_gtiff(path, dtype=None):
    ds = gdal.Open(path)
    if dtype is None:
        im_np = np.array(ds.ReadAsArray())
        return im_np.copy()
    else:
        im_np = np.array(ds.ReadAsArray(), dtype=dtype)
        return im_np.copy()


def load_bbox(path):
    with open(path) as f:
        y1, y2, x1, x2 = f.read().rstrip().split()
    return int(y1), int(y2), int(x1), int(x2)


def apply_homography(h, x):
    #                    h[0] h[1] h[2]
    # The convention is: h[3] h[4] h[5]
    #                    h[6] h[7] h[8]
    y = np.zeros_like(x)
    z = h[6]*x[0] + h[7]*x[1] + h[8]
    # tmp = x[0]

    y[0] = (h[0]*x[0] + h[1]*x[1] + h[2]) / z
    y[1] = (h[3]*x[0]  + h[4]*x[1] + h[5]) / z
    return y


def invert_homography(i):
    det = i[0]*i[4]*i[8] + i[2]*i[3]*i[7] + i[1]*i[5]*i[6] - i[2]*i[4]*i[6] - i[1]*i[3]*i[8] - i[0]*i[5]*i[7]
    o = np.zeros(9)
    o[0] = (i[4]*i[8] - i[5]*i[7]) / det
    o[1] = (i[2]*i[7] - i[1]*i[8]) / det
    o[2] = (i[1]*i[5] - i[2]*i[4]) / det
    o[3] = (i[5]*i[6] - i[3]*i[8]) / det
    o[4] = (i[0]*i[8] - i[2]*i[6]) / det
    o[5] = (i[2]*i[3] - i[0]*i[5]) / det
    o[6] = (i[3]*i[7] - i[4]*i[6]) / det
    o[7] = (i[1]*i[6] - i[0]*i[7]) / det
    o[8] = (i[0]*i[4] - i[1]*i[3]) / det
    return o

base_path = '/disk/songwei/LockheedMartion/end2end/'
mvs_path = os.path.join(base_path, 'MVS')
kml_path = os.path.join(base_path, 'KML')
gt_path = os.path.join(base_path, 'DSM')
filenames = os.listdir(mvs_path)

rsmes = [[], []]
accs = [[], []]
coms = [[], []]
l1es = [[], []]

for niter in range(1, 6):
    fw = open('debug/log%d.txt'%niter, 'w')
    for filename in filenames:
        fw.write(filename+'\n')
        tmp_path = os.path.join(mvs_path, filename)
        raw_path = '/disk/songwei/LockheedMartion/mvs_dataset/WV3/PAN'
        stereo_path = os.path.join(tmp_path, 'stereo/0/')
        img_path = os.path.join(tmp_path, 'cropped_images')
        if len(os.listdir(os.path.join(tmp_path, 'stereo'))) < 1:
            continue
        # load disparity map
        disp_file = os.path.join(stereo_path, 'out-F.tif')
        disparity_map = open_gtiff(disp_file, np.float32)

        # load bounding box
        bbox_file = os.path.join(tmp_path, 'bounds.txt')
        y1, y2, x1, x2 = load_bbox(bbox_file)

        # load homograph matrix
        exr_file_l = os.path.join(stereo_path, 'out-align-L.exr')
        exr_file_r = os.path.join(stereo_path, 'out-align-R.exr')
        h_left = OpenEXR.InputFile(exr_file_l)
        h_left = np.fromstring(h_left.channel('Channel0'), dtype = np.float32)
        h_left_inv = invert_homography(h_left)
        h_right = OpenEXR.InputFile(exr_file_r)
        h_right = np.fromstring(h_right.channel('Channel0'), dtype = np.float32)
        h_right_inv = invert_homography(h_right)

        # load TIFF image
        left_file = os.path.join(img_path, pairs_filenames[pair_id][0]+'.tif')
        left_stereo_file = os.path.join(stereo_path, 'out-L.tif')
        left_raw_file = os.path.join(raw_path, pairs_filenames[pair_id][0])
        right_file = os.path.join(img_path, pairs_filenames[pair_id][1]+'.tif')
        right_stereo_file = os.path.join(stereo_path, 'out-R.tif')
        right_raw_file = os.path.join(raw_path, pairs_filenames[pair_id][1])
        rpc_l = rpc_from_geotiff(left_file)
        rpc_l_raw = rpc_from_geotiff(left_raw_file)
        rpc_r = rpc_from_geotiff(right_file)
        rpc_r_raw = rpc_from_geotiff(right_raw_file)

        # load boundaries in original file
        data_left = open_gtiff(left_stereo_file)
        data_right = open_gtiff(right_stereo_file)

        data_left_un = open_gtiff(left_file)
        data_right_un = open_gtiff(right_file)

        nrow, ncol = data_left.shape
        height_map = np.zeros([3, data_left.shape[0], data_left.shape[1]])
        cu1, ru1 = np.meshgrid(np.arange(ncol, dtype=np.float64), np.arange(nrow, dtype=np.float64))

        ru2 = ru1 + disparity_map[1, :, :]
        cu2 = cu1 + disparity_map[0, :, :]

        cu1, ru1 = apply_homography(h_left_inv, [cu1, ru1])
        cu2, ru2 = apply_homography(h_right_inv, [cu2, ru2])
        # import ipdb;ipdb.set_trace()
        ru1 = ru1.reshape(-1)
        cu1 = cu1.reshape(-1)
        ru2 = ru2.reshape(-1)
        cu2 = cu2.reshape(-1)
        begin_time = time.time()
        Xu,Yu,Zu,error2d,error3d = triangulationRPC_matrix(ru1, cu1, ru2, cu2, rpc_l, rpc_r, Niter=niter, verbose=False)
        height_map[0, :, :] = Xu.reshape(height_map.shape[1], height_map.shape[2])
        height_map[1, :, :] = Yu.reshape(height_map.shape[1], height_map.shape[2])
        height_map[2, :, :] = Zu.reshape(height_map.shape[1], height_map.shape[2])

        fw.write('Triangulation took %.4f seconds\n'%(time.time()-begin_time))

        # pointclouds to DSM
        bounds, im_size = geo_utils.get_bounds_and_imsize_from_kml(os.path.join(kml_path, filename+'.kml'), 0.3)
        xx, yy = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
        wgs84 = pyproj.Proj('+proj=utm +zone=21 +datum=WGS84 +south')
        lons, lats = wgs84(height_map[0, :, :], height_map[1, :, :])
        ix, iy = geo_utils.spherical_to_image_positions(lons, lats, bounds, im_size)

        int_im = griddata((iy.reshape(-1), ix.reshape(-1)), height_map[2, :, :].reshape(-1), (yy, xx))
        int_im = geo_utils.fill_holes(int_im)

        fire_palette = scipy.misc.imread('image/fire_palette.png')[0][:, 0:3]
        color_map = eval_util.getColorMapFromPalette(int_im[y1:y2, x1:x2], fire_palette)
        scipy.misc.imsave('debug/triangulations/%s_out.png'%filename, color_map)

        amp_path = os.path.join(tmp_path, 'FF-0.npy')
        amp_data = np.load(amp_path)
        color_map = eval_util.getColorMapFromPalette(amp_data, fire_palette)
        scipy.misc.imsave('debug/triangulations/%s_amp.png'%filename, color_map)

        gt_data = np.load(os.path.join(gt_path, filename+'.npy'))
        color_map = eval_util.getColorMapFromPalette(gt_data, fire_palette)
        scipy.misc.imsave('debug/triangulations/%s_gt.png'%filename, color_map)

        rsme_asp, acc_asp, com_asp, l1e_asp = eval_util.evaluate(amp_data, gt_data)
        rsme_our, acc_our, com_our, l1e_our = eval_util.evaluate(int_im[y1:y2, x1:x2], gt_data)
        rsmes[0].append(rsme_asp)
        accs[0].append(acc_asp)
        coms[0].append(com_asp)
        l1es[0].append(l1e_asp)
        rsmes[1].append(rsme_our)
        accs[1].append(acc_our)
        coms[1].append(com_our)
        l1es[1].append(l1e_our)
        fw.write('The errors for asp are: rsme = %.4f, acc = %.4f, com = %.4f, l1e = %.4f\n'%(rsme_asp, acc_asp, com_asp, l1e_asp))
        fw.write('The errors for our are: rsme = %.4f, acc = %.4f, com = %.4f, l1e = %.4f\n'%(rsme_our, acc_our, com_our, l1e_our))


    fw.write('The average errors for asp are: rsme = %.4f +/- %.4f, acc = %.4f +/- %.4f, com = %.4f +/- %.4f, l1e = %.4f +/- %.4f\n'%(np.mean(rsmes[0]), 
        np.std(rsmes[0]), np.mean(accs[0]), np.std(accs[0]), np.mean(coms[0]), np.std(coms[0]), np.mean(l1es[0]), np.std(l1es[0])))
    fw.write('The average errors for our are: rsme = %.4f +/- %.4f, acc = %.4f +/- %.4f, com = %.4f +/- %.4f, l1e = %.4f +/- %.4f\n'%(np.mean(rsmes[1]), 
        np.std(rsmes[1]), np.mean(accs[1]), np.std(accs[1]), np.mean(coms[1]), np.std(coms[1]), np.mean(l1es[1]), np.std(l1es[1])))
