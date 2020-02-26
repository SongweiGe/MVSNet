"""
Data  Utils.
"""
try:
    import OpenEXR
    import imageio as misc
    from rpcm.rpc_model import rpc_from_geotiff
except :
    pass

import os
import time
import numpy as np
from .geo_utils import open_gtiff, get_bounds_and_imsize_from_kml, rpc_to_dict, RPCModel
import torch.utils.data as torch_data
from torchvision import transforms

def load_bbox(path):
    with open(path) as f:
        y1, y2, x1, x2 = f.read().rstrip().split()
    return int(y1), int(y2), int(x1), int(x2)


def save_height(outpath, hms, fns, mode):
    # hms: N x length x length
    for hm, fn in zip(hms, fns):
        np.save(os.path.join(os.path.join('../results', outpath, mode), fn+'.npy'), hm)


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

# load data from given file folder
def load_folder(tmp_path, kml_file, gt_path=None, mode='train', disp=False):
    # load input data
    stereo_path = os.path.join(tmp_path, 'stereo/0')
    img_path = os.path.join(tmp_path, 'cropped_images')

    # load TIFF image
    left_stereo_file = os.path.join(stereo_path, 'out-L.tif')
    right_stereo_file = os.path.join(stereo_path, 'out-R.tif')
    print(left_stereo_file)
    data_left = open_gtiff(left_stereo_file)
    data_left[data_left<0] = 0
    data_right = open_gtiff(right_stereo_file)
    data_right[data_right<0] = 0

    # load RPC model
    left_file = os.path.join(img_path, '02APR15WV031000015APR02134718-P1BS-500497282050_01_P001_________AAE_0AAAAABPABJ0.NTF.tif')
    right_file = os.path.join(img_path, '02APR15WV031000015APR02134802-P1BS-500276959010_02_P001_________AAE_0AAAAABPABC0.NTF.tif')
    rpc_l = rpc_from_geotiff(left_file)
    rpc_r = rpc_from_geotiff(right_file)

    # homography matrix
    exr_file_l = os.path.join(stereo_path, 'out-align-L.exr')
    exr_file_r = os.path.join(stereo_path, 'out-align-R.exr')
    h_left = OpenEXR.InputFile(exr_file_l)
    h_left = np.fromstring(h_left.channel('Channel0'), dtype = np.float32)
    h_left_inv = invert_homography(h_left)
    h_right = OpenEXR.InputFile(exr_file_r)
    h_right = np.fromstring(h_right.channel('Channel0'), dtype = np.float32)
    h_right_inv = invert_homography(h_right)

    # load bounding box
    bbox_file = os.path.join(tmp_path, 'bounds.txt')
    bbox = load_bbox(bbox_file) # y1, y2, x1, x2

    # load kml info
    bounds, im_size = get_bounds_and_imsize_from_kml(kml_file, 0.3)

    # load asp disparity map if True
    if disp:
        disp_file = os.path.join(stereo_path, 'out-F.tif')
        disparity_map = open_gtiff(disp_file, np.float32)[0, :, :]
    else:
        disparity_map = None

    if mode == 'train':
        # load ground truth
        height_gt = np.load(os.path.join(gt_path))
        return data_left, data_right, rpc_l, rpc_r, h_left_inv, h_right_inv, bbox, bounds, im_size, height_gt, disparity_map
    else:
        return data_left, data_right, rpc_l, rpc_r, h_left_inv, h_right_inv, bbox, bounds, im_size, disparity_map


class MVSdataset_raw(torch_data.Dataset):
    def __init__(self, gt_path, data_path, kml_path, img_size, filenames=None):
        # load input data
        self.img_pair = []
        self.rpc_pair = []
        self.h_pair = []
        self.area_info = []
        self.left_masks = []
        self.Ally = []
        self.pre_disps = []
        if filenames is None:
            filenames = os.listdir(data_path)
        # N x P x W x H x 2
        begin_time = time.time()
        print('start to load data')
        for filename in filenames:
            if len(os.listdir(os.path.join(data_path, filename, 'stereo'))) < 1:
                continue
            img_left, img_right, rpc_l, rpc_r, h_l, h_r, bbox, bounds, im_size, height_gt, disp = load_folder(tmp_path=os.path.join(data_path, filename), 
                        kml_file=os.path.join(kml_path, filename+'.kml'), gt_path=os.path.join(gt_path, filename+'.npy'), mode='train', disp=True)
            # import ipdb;ipdb.set_trace()
            img_left = np.pad(img_left, [[0, img_size-img_left.shape[0]], [0, img_size-img_left.shape[1]]], 'constant', constant_values=(0, 0))
            img_right = np.pad(img_right, [[0, img_size-img_right.shape[0]], [0, img_size-img_right.shape[1]]], 'constant', constant_values=(0, 0))
            disp = np.pad(disp, [[0, img_size-disp.shape[0]], [0, img_size-disp.shape[1]]], 'constant', constant_values=(0, 0))
            self.left_masks.append(img_left>0)
            self.img_pair.append(np.stack([img_left, img_right]))
            self.h_pair.append([h_l, h_r])
            self.rpc_pair.append([rpc_l, rpc_r])
            self.area_info.append([bbox, bounds, im_size])
            self.Ally.append(height_gt[:250, :250])
            self.pre_disps.append(disp)
        print('it took %.2fs to load data'%(time.time()-begin_time))

    def __getitem__(self, index):
        return self.img_pair[index], self.left_masks[index], self.h_pair[index], self.rpc_pair[index], self.area_info[index], self.pre_disps[index], self.Ally[index]

    def __len__(self):
        return len(self.h_pair)

    def save_data(self, filename='results/data_all.npz'):
        rpc_dict_pair = [[rpc_to_dict(item[0]), rpc_to_dict(item[1])] for item in self.rpc_pair]
        np.savez(filename, masks=self.left_masks, images=self.img_pair, hs=self.h_pair, rpcs=rpc_dict_pair, area_infos=self.area_info, disps=self.pre_disps, ys=self.Ally)
        data = np.load(filename, allow_pickle=True)


def data_split(n_total, filenames, boundary=33):
    test_ids = np.array([i for i, filename in enumerate(filenames) if int(filename.split('_')[2]) <= 33])
    train_ids = np.setdiff1d(np.arange(n_total), test_ids)
    return train_ids, test_ids


def get_numpy_dataset(filenames, bs, data_file='results/data_all.npz'):
    begin_time = time.time()
    # fetch the dataset
    data = np.load(data_file, allow_pickle=True)
    img_pair = data['images']
    # import ipdb;ipdb.set_trace()
    rpc_pair = np.array([[RPCModel(item[0]), RPCModel(item[1])] for item in data['rpcs']])
    h_pair = data['hs']
    area_info = data['area_infos']
    left_masks = data['masks'].astype(np.uint8)
    pre_disps = data['disps']
    Ally = data['ys']
    filenames = np.array(filenames)

    # split the dataset
    train_ids, test_ids = data_split(Ally.shape[0], filenames)

    train_dict = {'images': img_pair[train_ids], 'rpcs':rpc_pair[train_ids], 'hs': h_pair[train_ids], 
                'area_infos':area_info[train_ids], 'masks':left_masks[train_ids], 'disps':pre_disps[train_ids], 
                'ys':Ally[train_ids], 'names':filenames[train_ids]}
    test_dict = {'images': img_pair[test_ids], 'rpcs':rpc_pair[test_ids], 'hs': h_pair[test_ids], 
                'area_infos':area_info[test_ids], 'masks':left_masks[test_ids], 'disps':pre_disps[test_ids], 
                'ys':Ally[test_ids], 'names':filenames[test_ids]}
    train_dataset = MVSdataset(train_dict)
    test_dataset = MVSdataset(test_dict)
    print('it took %.2fs to load data'%(time.time()-begin_time))
    return torch_data.DataLoader(train_dataset, batch_size=bs, shuffle=False, pin_memory=True), \
            torch_data.DataLoader(test_dataset, batch_size=bs, pin_memory=True)


class MVSdataset(torch_data.Dataset):
    def __init__(self, data):
        # load input data
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.img_pair = data['images']
        self.rpc_pair = data['rpcs']
        self.h_pair = data['hs']
        self.area_info = data['area_infos']
        self.left_masks = data['masks']
        self.pre_disps = data['disps']
        self.Ally = data['ys']
        self.names = data['names']
    def __getitem__(self, index):
        # import ipdb;ipdb.set_trace()
        area_info = np.array(list(self.area_info[index][0]) + self.area_info[index][1][0] +
                            self.area_info[index][1][1] + list(self.area_info[index][2]))
        rpc_list = np.array([self.rpc_pair[index][0].to_list(), self.rpc_pair[index][1].to_list()])
        elem = {'images': self.img_pair[index], 
                'hs': self.h_pair[index], 
                'area_infos':area_info, 
                'masks':self.left_masks[index], 
                'disps':self.pre_disps[index], 
                'ys':self.Ally[index],
                'names':self.names[index], 
                'rpcs':rpc_list
                }
        return elem
    def __len__(self):
        return len(self.h_pair)


class MVSdataset_lithium(torch_data.Dataset):
    def __init__(self, data_file='results/data_all.npz'):
        # load input data
        begin_time = time.time()
        data = np.load(data_file, allow_pickle=True)
        self.img_pair = data['images']
        self.rpc_pair = [[RPCModel(item[0]), RPCModel(item[1])] for item in data['rpcs']]
        self.h_pair = data['hs']
        self.area_info = data['area_infos']
        self.left_masks = data['masks']
        self.pre_disps = data['disps']
        self.Ally = data['ys']
        print('it took %.2fs to load data'%(time.time()-begin_time))
    def __getitem__(self, index):
        return self.img_pair[index], self.left_masks[index], self.h_pair[index], self.rpc_pair[index], self.area_info[index], self.pre_disps[index], self.Ally[index]
    def __len__(self):
        return len(self.h_pair)


if __name__ == '__main__':
    data_path = '/disk/songwei/LockheedMartion/end2end/MVS/'
    kml_path = '/disk/songwei/LockheedMartion/DeepVote/kml/'
    gt_path = '/disk/songwei/LockheedMartion/DeepVote/DSM/'
    # AllX, Ally = load_data(gt_path, data_path, kml_path)
    train_dataset= MVSdataset(gt_path, data_path, kml_path)
    train_loader = torch_data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    import ipdb;ipdb.set_trace()