import os
import gdal
import OpenEXR
import numpy as np
from rpcm.rpc_model import rpc_from_geotiff
from triangulationRPC import triangulationRPC

disparity_marigin = 288

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
    y = np.zeros(2, dtype=np.float)
    z = h[6]*x[0] + h[7]*x[1] + h[8]
    tmp = x[0]
    y[0] = (h[0]*x[0] + h[1]*x[1] + h[2]) / z
    y[1] = (h[3]*tmp  + h[4]*x[1] + h[5]) / z
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


tmp_path = '/home/songwei/LM/iarpa_contest_submission/tmp/'
raw_path = '/disk/songwei/LockheedMartion/mvs_dataset/WV3/PAN'
stereo_path = os.path.join(tmp_path, 'stereo_output/0/results')
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
left_file = os.path.join(img_path, '02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF.tif')
left_raw_file = os.path.join(raw_path, '02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF')
right_file = os.path.join(img_path, '02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.NTF.tif')
right_raw_file = os.path.join(raw_path, '02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.NTF')
rpc_l = rpc_from_geotiff(left_file)
rpc_l_raw = rpc_from_geotiff(left_raw_file)
rpc_r = rpc_from_geotiff(right_file)
rpc_r_raw = rpc_from_geotiff(right_raw_file)

# load boundaries in original file
left_bound = np.load(left_file+'.npy')
right_bound = np.load(right_file+'.npy')

# sanity check rpc
fn = '02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF.tif'
rpc_ls = [rpc_from_geotiff(os.path.join('/disk/songwei/LockheedMartion/end2end/MVS/dem_2_%d/cropped_images'%i, fn)) for i in range(1,5)]
# data = open_gtiff(left_file)
# data = open_gtiff(left_raw_file)
# data[left_bound[1]:left_bound[1]+10, left_bound[0]:left_bound[0]+10]

height_map = np.zeros((y2-y1, x2-x1))
for row in range(y1, y2):
    for col in range(x1, x2):
        # if disparity_map[2, row, col] == 0:
        #     continue
        left_row_un, left_col_un = apply_homography(h_left_inv, [row, col-disparity_marigin])
        right_row_un, right_col_un = apply_homography(h_right_inv, [row+disparity_map[0, row, col], col-disparity_marigin+disparity_map[1, row, col]])
        # import ipdb;ipdb.set_trace()
        # triangulationRPC(left_row_un+left_bound[1], left_col_un+left_bound[0], right_row_un+right_bound[1], right_col_un+right_bound[0], rpc_l_raw, rpc_r_raw, verbose=True)
        triangulationRPC(0, 0, 0, 0, rpc_l, rpc_r, verbose=True)
        height_map[row-y1, col-x1]


import ipdb;ipdb.set_trace()