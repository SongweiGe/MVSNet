import os
import cv2
import scipy.misc
import gdal
import OpenEXR
import numpy as np
import matplotlib.pyplot as plt
from rpcm.rpc_model import rpc_from_geotiff
from triangulationRPC import triangulationRPC


pairs_filenames = [
            ['02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF', 
            '02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.NTF'], 
            ['02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF', 
            '02APR15WV031000015APR02134804-P1BS-500497284080_01_P001_________AAE_0AAAAABPABJ0.NTF'], 
            ['02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.NTF', 
            '02APR15WV031000015APR02134716-P1BS-500276959010_02_P001_________AAE_0AAAAABPABB0.NTF'], 
            ['02APR15WV031000015APR02134804-P1BS-500497284080_01_P001_________AAE_0AAAAABPABJ0.NTF', 
            '02APR15WV031000015APR02134716-P1BS-500276959010_02_P001_________AAE_0AAAAABPABB0.NTF'], 
            ['02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF', 
            '03APR15WV031000015APR03140238-P1BS-500497283030_01_P001_________AAE_0AAAAABPABR0.NTF'], 
            ['22MAR15WV031000015MAR22141208-P1BS-500497285090_01_P001_________AAE_0AAAAABPABQ0.NTF', 
            '21MAR15WV031000015MAR21135704-P1BS-500497282060_01_P001_________AAE_0AAAAABPABQ0.NTF'], 
            ['14SEP15WV031000015SEP14140305-P1BS-500497285020_01_P001_________AAE_0AAAAABPABS0.NTF', 
            '15SEP15WV031000015SEP15141840-P1BS-500497285060_01_P001_________AAE_0AAAAABPABO0.NTF'], 
            ['22OCT15WV031000015OCT22140432-P1BS-500497282010_01_P001_________AAE_0AAAAABPABS0.NTF', 
            '23OCT15WV031100015OCT23141928-P1BS-500497285030_01_P001_________AAE_0AAAAABPABO0.NTF'], 
            ['19JUN15WV031100015JUN19141753-P1BS-500346924040_01_P001_________AAE_0AAAAABPABP0.NTF', 
            '18JUN15WV031000015JUN18140207-P1BS-500497285040_01_P001_________AAE_0AAAAABPABR0.NTF'], 
            ['11FEB15WV031000015FEB11135123-P1BS-500497282030_01_P001_________AAE_0AAAAABPABR0.NTF', 
            '12FEB15WV031000015FEB12140652-P1BS-500497283100_01_P001_________AAE_0AAAAABPABQ0.NTF'], 
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
    y = np.zeros(2, dtype=np.float)
    z = h[6]*x[0] + h[7]*x[1] + h[8]
    # tmp = x[0]
    y[0] = (h[0]*x[0] + h[1]*x[1] + h[2]) / z
    y[1] = (h[3]*x[0]  + h[4]*x[1] + h[5]) / z
    return y


# def apply_homography(h, x):
#     #                    h[0] h[1] h[2]
#     # The convention is: h[3] h[4] h[5]
#     #                    h[6] h[7] h[8]
#     y = np.zeros(2, dtype=np.float)
#     y[0] = h[0]*x[0] + h[1]*x[1] + h[2]
#     y[1] = h[3]*x[0]  + h[4]*x[1] + h[5]
#     return y


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
stereo_path = os.path.join(tmp_path, 'stereo_output/%d/results'%pair_id)
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

# sanity check rpc
# fn = '02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF.tif'
# rpc_ls = [rpc_from_geotiff(os.path.join('/disk/songwei/LockheedMartion/end2end/MVS/dem_2_%d/cropped_images'%i, fn)) for i in range(1,5)]
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

# # y1_raw, x1_raw = apply_homography(h_left_inv, [y1, x1])
# x2_raw, y2_raw = apply_homography(h_left_inv, [300, 500])
# cv2.perspectiveTransform(np.float32([[[300, 500]]]), h_left_inv.reshape((3,3)))
# scipy.misc.imsave('left_raw_debug.png', data_left_un[int(y2_raw)-100:int(y2_raw)+100, int(x2_raw)-100:int(x2_raw)+100])
# # scipy.misc.imsave('left_raw_debug.png', data_left_un[int(y1_raw)-150:int(y1_raw)+150, int(x1_raw)-150:int(x1_raw)+150])

# # y1_raw, x1_raw = apply_homography(h_right_inv, [y1, x1])
# x2_raw, y2_raw  = apply_homography(h_right_inv, [300, 500-disparity_map[0, 300, 500]])
# scipy.misc.imsave('right_raw_debug.png', data_right_un[int(y2_raw)-100:int(y2_raw)+100, int(x2_raw)-100:int(x2_raw)+100])
# # scipy.misc.imsave('right_raw_debug.png', data_right_un[int(y1_raw)-150:int(y1_raw)+150, int(x1_raw)-150:int(x1_raw)+150])

# img_rec_left = cv2.warpPerspective(data_left_un, h_left.reshape((3,3)), (data_right.shape[1],data_right.shape[0]))
# scipy.misc.imsave('left_recti.png',img_rec_left)
# img_unrec_left = cv2.warpPerspective(img_rec_left, h_left_inv.reshape((3,3)), (data_left_un.shape[1],data_left_un.shape[0]))
# scipy.misc.imsave('left_recti_unrec.png',img_unrec_left)
# aaa = np.matmul(h_left_inv.reshape((3,3)), np.array([y2, x2, 1]))
# np.matmul(h_left_inv.reshape((3,3)), aaa)

# img_rec_right = cv2.warpPerspective(data_right_un, h_right.reshape((3,3)), (data_right.shape[1],data_right.shape[0]))
# scipy.misc.imsave('right_recti.png',img_rec_right)
# img_unrec_right = cv2.warpPerspective(img_rec_right, h_right_inv.reshape((3,3)), (data_right_un.shape[1],data_right_un.shape[0]))
# scipy.misc.imsave('right_recti_unrec.png',img_unrec_right)

# xs = []
# ys = []
# for i in range(data_right.shape[0]):
#     for j in range(data_right.shape[1]):
#         if data_right[i, j] == 0:
#             continue
#         x, y = apply_homography(h_right_inv, [j, i])
#         xs.append(x)
#         ys.append(y)

# np.max(xs), np.min(xs)
# np.max(ys), np.min(ys)

# scipy.misc.imsave('disparity_map.png',disparity_map[0, :, :])
# data_left_raw = open_gtiff(left_raw_file)
# data_right_raw = open_gtiff(right_raw_file)
# (data_left_un[y1:y2, x1:x2] == data_left_raw[y1+left_bound[1]:y2+left_bound[1], x1+left_bound[0]:x2+left_bound[0]]).all()


###### debug rpc #######
# rpc_param_1 = np.loadtxt('rpc1.out', delimiter=',')
# rpc_param_2 = np.loadtxt('rpc2.out', delimiter=',')
# globalposc1 = rpc_param_1[95-1];
# globalposr1 = rpc_param_1[96-1];
# globalposc2 = rpc_param_2[95-1];
# globalposr2 = rpc_param_2[96-1];        
# p1_1=rpc_param_1[11-1:30];p2_1=rpc_param_1[31-1:50];
# p3_1=rpc_param_1[51-1:70];p4_1=rpc_param_1[71-1:90];
# p1_2=rpc_param_2[11-1:30];p2_2=rpc_param_2[31-1:50];
# p3_2=rpc_param_2[51-1:70];p4_2=rpc_param_2[71-1:90];
# scale_offsets_1 = [rpc_param_1[8-1], rpc_param_1[3-1],
#     rpc_param_1[9-1], rpc_param_1[4-1], rpc_param_1[10-1], rpc_param_1[5-1],
#     rpc_param_1[1-1], rpc_param_1[2-1], rpc_param_1[6-1], rpc_param_1[7-1]];
# scale_offsets_2 = [rpc_param_2[8-1], rpc_param_2[3-1],
#     rpc_param_2[9-1], rpc_param_2[4-1], rpc_param_2[10-1], rpc_param_2[5-1],
#     rpc_param_2[1-1], rpc_param_2[2-1], rpc_param_2[6-1], rpc_param_2[7-1]];
# triangulationRPC(p1_1,p2_1,p3_1,p4_1,p1_2,p2_2,p3_2,p4_2,2.1279e+04,1.7762e+04,2.0001e+04,1.7510e+04,scale_offsets_1,scale_offsets_2, verbose=True)
# items_l = [rpc_l_raw.lon_scale, rpc_l_raw.lon_offset, rpc_l_raw.lat_scale, rpc_l_raw.lat_offset, rpc_l_raw.alt_scale, rpc_l_raw.alt_offset, rpc_l_raw.row_scale, rpc_l_raw.row_offset, rpc_l_raw.col_scale, rpc_l_raw.col_offset]
# items_l += rpc_l_raw.row_num + rpc_l_raw.row_den + rpc_l_raw.col_num + rpc_l_raw.col_den
# items_l += [0, 0, 0, 0, left_bound[1], left_bound[0]]
# with open('rpc1.txt', 'w') as fw: fw.write(', '.join([str(item) for item in items_l]))

# items_r = [rpc_r_raw.lon_scale, rpc_r_raw.lon_offset, rpc_r_raw.lat_scale, rpc_r_raw.lat_offset, rpc_r_raw.alt_scale, rpc_r_raw.alt_offset, rpc_r_raw.row_scale, rpc_r_raw.row_offset, rpc_r_raw.col_scale, rpc_r_raw.col_offset]
# items_r += rpc_r_raw.row_num + rpc_r_raw.row_den + rpc_r_raw.col_num + rpc_r_raw.col_den
# items_r += [0, 0, 0, 0, right_bound[1], right_bound[0]]
# with open('rpc2.txt', 'w') as fw: fw.write(', '.join([str(item) for item in items_r]))

import ipdb;ipdb.set_trace()
# height_map = np.zeros((y2-y1, x2-x1))
height_map = np.zeros_like(data_left_un)
for row in range(y1, y2):
    for col in range(x1, x2):
        # if disparity_map[2, row, col] == 0:
        #     continue
        left_col_un, left_row_un = apply_homography(h_left_inv, [col, row])
        right_col_un, right_row_un = apply_homography(h_right_inv, [col+disparity_map[0, row, col], row+disparity_map[1, row, col]])
        # scipy.misc.imsave('left_raw_debug.png', data_left_un[int(left_col_un)-100:int(left_col_un)+100, int(left_row_un)-100:int(left_row_un)+100])
        # scipy.misc.imsave('right_raw_debug.png', data_right_un[int(right_col_un)-100:int(right_col_un)+100, int(right_row_un)-100:int(right_row_un)+100])
        # triangulationRPC(left_row_un+left_bound[1], left_col_un+left_bound[0], right_row_un+right_bound[1], right_col_un+right_bound[0], rpc_l_raw, rpc_r_raw, verbose=True)
        Xu,Yu,Zu,error2d,error3d = triangulationRPC(left_row_un+left_bound[1], left_col_un+left_bound[0], right_row_un+right_bound[1], right_col_un+right_bound[0], rpc_l_raw, rpc_r_raw, verbose=False)
        # Xu,Yu,Zu,error2d,error3d = triangulationRPC(left_row_un, left_col_un, right_row_un, right_col_un, rpc_l, rpc_r, verbose=True)
        # triangulationRPC(row, col, row, col, rpc_l, rpc_r, verbose=True)
        height_map[row-y1, col-x1] = Zu

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

import ipdb;ipdb.set_trace()


amp_path = os.path.join(tmp_path, 'FF-%d.npy'%pair_id)
amp_data = np.load(amp_path)[y1:y2, x1:x2]
fig1, ax1 = plt.subplots(1, 1)
cs = ax1.imshow(amp_data, cmap='Reds')
fig1.colorbar(cs)
plt.savefig('hm_gt.png')

amp_path = os.path.join(stereo_path, 'out-PC.tif')
amp_data = open_gtiff(amp_path)
plt.close()
fig1, ax1 = plt.subplots(1, 1)
cs = ax1.imshow(amp_data[0, :, :], cmap='Reds')
fig1.colorbar(cs)
plt.savefig('amp.png')
height_map2 = height_map-height_map.min()+amp_data.min()
scipy.misc.imsave('height_map.png', height_map2)
# scipy.misc.imsave('color_map.png', data_left[y1:y2, x1:x2])

import matplotlib; matplotlib.pyplot.switch_backend('agg')
fig1, ax1 = plt.subplots(1, 1)
cs = ax1.imshow(height_map, cmap='Reds')
fig1.colorbar(cs)
plt.savefig('hm.png')

