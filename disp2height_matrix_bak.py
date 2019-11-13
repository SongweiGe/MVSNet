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

base_path = '/disk/songwei/LockheedMartion/end2end/MVS'
filenames = os.listdir(base_path)

filename = filenames[0]
print(filename)
tmp_path = os.path.join(base_path, filename)
raw_path = '/disk/songwei/LockheedMartion/mvs_dataset/WV3/PAN'
stereo_path = os.path.join(tmp_path, 'stereo/0/')
img_path = os.path.join(tmp_path, 'cropped_images')

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
left_bound = np.load(left_file+'.npy')
right_bound = np.load(right_file+'.npy')

data_left = open_gtiff(left_stereo_file)
# data_left[data_left<0] = 0
data_right = open_gtiff(right_stereo_file)
# data_right[data_right<0] = 0
# scipy.misc.imsave('left_debug.png', data_left)
# scipy.misc.imsave('right_debug.png', data_right)

data_left_un = open_gtiff(left_file)
# scipy.misc.imsave('left_un.png', data_left_un)
data_right_un = open_gtiff(right_file)
# scipy.misc.imsave('right_un.png', data_right_un)

nrow, ncol = data_left.shape
height_map = np.zeros([3, data_left.shape[0], data_left.shape[1]])
# f_row = interp2d(np.arange(disparity_map.shape[2]), np.arange(disparity_map.shape[1]), disparity_map[1, :, :])
# f_col = interp2d(np.arange(disparity_map.shape[2]), np.arange(disparity_map.shape[1]), disparity_map[0, :, :])
cu1, ru1 = np.meshgrid(np.arange(ncol, dtype=np.float64), np.arange(nrow, dtype=np.float64))

ru2 = ru1 + disparity_map[1, :, :]
cu2 = cu1 + disparity_map[0, :, :]

cu1, ru1 = apply_homography(h_left_inv, [cu1, ru1])
cu2, ru2 = apply_homography(h_right_inv, [cu2, ru2])
# import ipdb;ipdb.set_trace()
# ru1 = (ru1+left_bound[1]).reshape(-1)
# cu1 = (cu1+left_bound[0]).reshape(-1)
# ru2 = (ru2+right_bound[1]).reshape(-1)
# cu2 = (cu2+right_bound[0]).reshape(-1)
ru1 = ru1.reshape(-1)
cu1 = cu1.reshape(-1)
ru2 = ru2.reshape(-1)
cu2 = cu2.reshape(-1)
begin_time = time.time()
# Xu,Yu,Zu,error2d,error3d = triangulationRPC_array(ru1, cu1, ru2, cu2, rpc_l_raw, rpc_r_raw, verbose=False)
Xu,Yu,Zu,error2d,error3d = triangulationRPC_matrix(ru1, cu1, ru2, cu2, rpc_l, rpc_r, verbose=False)
height_map[0, :, :] = Xu.reshape(height_map.shape[1], height_map.shape[2])
height_map[1, :, :] = Yu.reshape(height_map.shape[1], height_map.shape[2])
height_map[2, :, :] = Zu.reshape(height_map.shape[1], height_map.shape[2])

print('Triangulation took %.4f seconds'%(time.time()-begin_time))
# row = y1+300
# col = x1+300
# left_row_un, left_col_un = apply_homography(h_left_inv, [row, col])
# right_row_un, right_col_un = apply_homography(h_right_inv, [row+disparity_map[1, row, col], col+disparity_map[0, row, col]])
# scipy.misc.imsave('left_raw_debug.png', data_left_un[int(left_col_un)-100:int(left_col_un)+100, int(left_row_un)-100:int(left_row_un)+100])
# scipy.misc.imsave('right_raw_debug.png', data_right_un[int(right_col_un)-100:int(right_col_un)+100, int(right_row_un)-100:int(right_row_un)+100])
# scipy.misc.imsave('left_raw2_debug.png', data_left_raw[int(left_col_un+left_bound[1])-100:int(left_col_un+left_bound[1])+100, int(left_row_un+left_bound[0])-100:int(left_row_un+left_bound[0])+100])
# scipy.misc.imsave('right_raw2_debug.png', data_right_raw[int(right_col_un+right_bound[1])-100:int(right_col_un+right_bound[1])+100, int(right_row_un+right_bound[0])-100:int(right_row_un+right_bound[0])+100])

# left_row_un, left_col_un = apply_homography(h_left, [row, col])
# right_row_un, right_col_un = apply_homography(h_right, [row+disparity_map[1, row, col], col+disparity_map[0, row, col]])
# Xu,Yu,Zu,error2d,error3d = triangulationRPC(left_row_un+left_bound[1], left_col_un+left_bound[0], right_row_un+right_bound[1], right_col_un+right_bound[0], rpc_l_raw, rpc_r_raw, verbose=True)


# pointclouds to DSM
bounds, im_size = geo_utils.get_bounds_and_imsize_from_kml('/disk/songwei/LockheedMartion/DeepVote/kml/dem_4_23.kml', 0.3)
xx, yy = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
wgs84 = pyproj.Proj('+proj=utm +zone=21 +datum=WGS84 +south')
lons, lats = wgs84(height_map[0, :, :], height_map[1, :, :])
ix, iy = geo_utils.spherical_to_image_positions(lons, lats, bounds, im_size)
import ipdb;ipdb.set_trace()

int_im = griddata((iy.reshape(-1), ix.reshape(-1)), height_map[2, :, :].reshape(-1), (yy, xx))
int_im = geo_utils.fill_holes(int_im)

fire_palette = scipy.misc.imread('image/fire_palette.png')[0][:, 0:3]
color_map = eval_util.getColorMapFromPalette(int_im[y1:y2, x1:x2], fire_palette)
scipy.misc.imsave('final_matrix_localrpc.png', color_map)


amp_path = os.path.join(tmp_path, 'FF-%d.npy'%pair_id)
amp_data = np.load(amp_path)
color_map = eval_util.getColorMapFromPalette(amp_data[y1:y2, x1:x2], fire_palette)
scipy.misc.imsave('final_amp_matrix_localrpc.png', color_map)

amp_path = os.path.join(stereo_path, 'out-PC.tif')
amp_data = open_gtiff(amp_path)

color_map = eval_util.getColorMapFromPalette(height_map[2, :, :], fire_palette)
scipy.misc.imsave('height_map_matrix_localrpc.png', color_map)
