import os
import time
import imageio
import pyproj
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import utils.data_util as data_util
from utils.model_util import FlowNetS
from scipy.interpolate import griddata

from utils import geo_utils, eval_util
from triangulationRPC_matrix_torch import triangulationRPC_matrix
# from torchinterp1d import Interp1d

fire_palette = imageio.imread('./image/fire_palette.png')[0][:, 0:3]

class Trainer(object):
    def __init__(self, args, dataloader):
        self.args = args
        self.n_folds = args.n_folds
        self.n_epochs = args.epochs
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.D = FlowNetS(input_channels=3).cuda()
        # self.L = nn.MSELoss().cuda()
        self.L = nn.L1Loss().cuda()
        self.optimizer = torch.optim.Adadelta(self.D.parameters(), lr=1e-0)
        self.dataloader = dataloader
        self.n_total = len(self.dataloader)
        self.shuffled_index = np.arange(self.n_total)
        np.random.seed(2019)
        np.random.shuffle(self.shuffled_index)
        self.wgs84 = pyproj.Proj('+proj=utm +zone=21 +datum=WGS84 +south')
        # self.interp_method = Interp1d()
        self.cu1_rec, self.ru1_rec = torch.meshgrid([torch.arange(self.img_size), torch.arange(self.img_size)])
        self.cu1_rec = self.cu1_rec.type(torch.cuda.DoubleTensor).transpose(1, 0)
        self.ru1_rec = self.ru1_rec.type(torch.cuda.DoubleTensor).transpose(1, 0)
        self.xx, self.yy = np.meshgrid(np.arange(250), np.arange(250))


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
        cu2_rec = self.cu1_rec + disparity_map[0, 0, :, :].type(torch.cuda.DoubleTensor)
        ru2_rec = self.ru1_rec
        # cu2_rec = cu1_rec

        cu1, ru1 = self.apply_homography(h_left_inv, [self.cu1_rec, self.ru1_rec])
        cu2, ru2 = self.apply_homography(h_right_inv, [cu2_rec, ru2_rec])
        ru1 = ru1[masks].reshape(-1)
        cu1 = cu1[masks].reshape(-1)
        ru2 = ru2[masks].reshape(-1)
        cu2 = cu2[masks].reshape(-1)
        # import ipdb;ipdb.set_trace()
        Xu, Yu, Zu, _, _ = triangulationRPC_matrix(ru1, cu1, ru2, cu2, rpc_l, rpc_r, verbose=False, inverse_bs=64000)

        lons, lats = self.wgs84(Xu.cpu().data.numpy(), Yu.cpu().data.numpy())
        ix, iy = geo_utils.spherical_to_image_positions(lons, lats, bounds, im_size)
        valid_points = np.logical_and(np.logical_and(iy>bbox[0], iy<bbox[0]+250), np.logical_and(ix>bbox[2], ix<bbox[2]+250))
        # input_coords = torch.stack([torch.cuda.FloatTensor(iy), torch.cuda.FloatTensor(ix)])
        # output_coords = torch.stack([torch.cuda.FloatTensor(self.yy), torch.cuda.FloatTensor(self.xx)])
        # int_im = self.interp_method(input_coords, Zu[valid_points], output_coords)
        # int_im = griddata((iy, ix), Zu[valid_points], (self.yy, self.xx))
        # int_im = geo_utils.fill_holes(int_im)
        # return int_im[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        return ix[valid_points]-bbox[2], iy[valid_points]-bbox[0], Zu[np.where(valid_points)], Xu, Yu, Zu

    def calculate_loss(self, X, Y, Z, gt):
        x_max, y_max = gt.shape[1:]
        # [-1, 1]
        grid = torch.stack([torch.cuda.FloatTensor(X)/x_max*2-1, torch.cuda.FloatTensor(Y)/y_max*2-1]).transpose(1, 0).view(1, 1, -1, 2)
        gt_height = torch.nn.functional.grid_sample(gt.unsqueeze(1), grid.cuda()).squeeze()
        # import ipdb;ipdb.set_trace()

        # xx, yy = np.meshgrid(np.arange(250), np.arange(250))
        # input_coords = torch.stack([torch.cuda.FloatTensor(Y), torch.cuda.FloatTensor(X)])
        # output_coords = torch.stack([torch.cuda.FloatTensor(yy), torch.cuda.FloatTensor(xx)])
        # int_im = griddata((Y, X), Z.cpu().data.numpy(), (yy, xx))
        # int_im = geo_utils.fill_holes(int_im)

        # int_gt = griddata((Y, X), gt_height.cpu().data.numpy(), (yy, xx))
        # int_gt = geo_utils.fill_holes(int_gt)


        # debug_path = 'debug/loss'
        # color_map = eval_util.getColorMapFromPalette(gt[0].cpu().data.numpy(), fire_palette)
        # imageio.imsave(os.path.join(debug_path, 'gt.png'), color_map)
                
        # color_map = eval_util.getColorMapFromPalette(int_gt, fire_palette)
        # imageio.imsave(os.path.join(debug_path, 'recon_gt.png'), color_map)

        # color_map = eval_util.getColorMapFromPalette(int_im, fire_palette)
        # imageio.imsave(os.path.join(debug_path, 'pred.png'), color_map)
        return self.L(gt_height, Z.cuda())

    def data_split(self, fold_id):
        fold_size = self.n_total//self.n_folds
        if self.n_folds > 1:
            test_ids = self.shuffled_index[list(range(i*fold_size, (i+1)*fold_size))]
            # train_ids = self.shuffled_index
            train_ids = np.setdiff1d(self.shuffled_index, test_ids)
        else: # split according to the geographical locations
            test_ids = np.array([i for i, filename in enumerate(filenames) if int(filename.split('_')[2]) <= 33])
            # import ipdb;ipdb.set_trace()
            train_ids = np.setdiff1d(self.shuffled_index, test_ids)
        return train_ids, test_ids

    def run(self):
        self.D.train()
        loss_val = float('inf')
        self.train_loss = []
        outputs = []
        cv_losses = []
        fold_size = self.n_total//self.n_folds
        # 5 cross validation
        print('start training')
        for i in range(1):
        # for i in range(self.n_folds):
            train_ids, test_ids = self.data_split(i)
            # test_ids = self.shuffled_index[list(range(i*fold_size, (i+1)*fold_size))]
            # train_ids = self.shuffled_index
            # import ipdb;ipdb.set_trace()
            n_batch = train_ids.shape[0]//self.batch_size-1
            n_test_batch = test_ids.shape[0]//self.batch_size-1
            for p in self.D.parameters():
                self.weights_init(p)
            for j in range(self.n_epochs):
                np.random.shuffle(train_ids)
                begin = time.time()
                train_epoch_loss = []
                test_epoch_loss = []
                # import ipdb;ipdb.set_trace()
                for k in range(n_batch+1):
                    #forward calculation and back propagation, X: B x P x 2 x W x H
                    if k == n_batch:
                        train_batch_ids = train_ids[k*self.batch_size:]
                    else:
                        train_batch_ids = train_ids[(k+0)*self.batch_size:(k+1)*self.batch_size]
                    img_pair, masks, h_pair, rpc_pair, area_info, pre_disp, y = self.dataloader.__getitem__(train_batch_ids[0])

                    X = Variable(torch.cuda.FloatTensor([np.vstack([img_pair, np.expand_dims(pre_disp, 0)])]), requires_grad=False)
                    Y = Variable(torch.cuda.FloatTensor([y]), requires_grad=False)
                    Disp = Variable(torch.cuda.FloatTensor(np.expand_dims(np.expand_dims(pre_disp, 0), 0)), requires_grad=False)
                    h_pair = torch.cuda.DoubleTensor(np.stack(h_pair))
                    masks = torch.tensor(masks.tolist())
                    self.optimizer.zero_grad()
                    # import ipdb;ipdb.set_trace()
                    
                    disparity_map = self.D(X)[0]+Disp
                    lon, lat, heights, Xu, Yu, Zu = self.triangulation_forward(disparity_map, masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
                    print('range of predicted height: (%.3f, %.3f), ground truth: (%.3f, %.3f)'%(heights.min(), heights.max(), Y.min(), Y.max()))
                    loss = self.calculate_loss(lon, lat, heights, Y)
                    loss_val = loss.data.cpu().numpy()
                    print('The number of remaining points:%d'%len(lon))
                    if np.isnan(loss_val):
                        continue
                        # import ipdb;ipdb.set_trace()
                    loss.backward()
                    self.optimizer.step()
                    train_epoch_loss.append(loss_val)
                    del X,Y,lon, lat, heights,loss
                    print("Epochs %d, iteration: %d, time = %ds, training loss: %f"%(i, j, time.time() - begin, loss_val))
                
                if (j+1)%5 == 0:
                    torch.save(self.D.state_dict(), os.path.join('results', self.args.exp_name, 'models', 'fold%d_%d'%(i, j)))
                print("Fold %d, Epochs %d, time = %ds, training loss: %f"%(i, j, time.time() - begin, np.mean(train_epoch_loss)))
            
            # save the last training estimation
            if self.args.save_train:
                output = (pred_height.cpu().data.numpy())
                data_util.save_height(self.args.exp_name, output, filenames[train_batch_ids], 'train')
            # test
            for k in range(n_test_batch+1):
                if k == n_test_batch:
                    test_batch_ids = test_ids[k*self.batch_size:]
                else:
                    test_batch_ids = test_ids[k*self.batch_size:(k+1)*self.batch_size]
                img_pair, masks, h_pair, rpc_pair, area_info, pre_disp, y = self.dataloader.__getitem__(train_batch_ids[0])
                X = Variable(torch.cuda.FloatTensor([np.stack(img_pair, pre_disp)]), requires_grad=False)
                Y = Variable(torch.cuda.FloatTensor([y]), requires_grad=False)
                Disp = Variable(torch.cuda.FloatTensor(pre_disp), requires_grad=False)
                h_pair = torch.cuda.DoubleTensor(np.stack(h_pair))
                masks = torch.tensor(masks.tolist())
                disparity_map = self.D(X)[0]+Disp
                # lon, lat, heights, Xu, Yu, Zu = self.triangulation_forward(disparity_map[:, :, 300:400, 300:400], masks[300:400, 300:400], 100, 100, rpc_pair, h_pair, area_info)
                lon, lat, heights, Xu, Yu, Zu = self.triangulation_forward(disparity_map, masks, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
                loss = self.calculate_loss(lon, lat, heights, Y)
                loss_val = loss.data.cpu().numpy()
                test_epoch_loss.append(loss_val)
                output = (pred_height.cpu().data.numpy())
                data_util.save_height(self.args.exp_name, output, filenames[test_batch_ids], 'test')
                del X,Y,pred_height,loss
            print("Fold %d, Epochs %d, time = %ds, training loss: %f, test loss %f"%(i, j, time.time() - begin, np.mean(train_epoch_loss), np.mean(test_epoch_loss)))

            cv_losses.append(np.mean(test_epoch_loss))
        print('overall performance: %f'%np.mean(cv_losses))
        return cv_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, default='plain', help='the name to identify current experiment')
    parser.add_argument("-ie", "--input_epoch", type=str, default=None, help='Load model after n epochs')
    parser.add_argument("-ip", "--input_fold", type=str, default='0', help='Load model filepath')
    parser.add_argument("-ld", "--load_model", type=bool, default=False, help='Load pretrained model or not')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    parser.add_argument('-nf', '--n_folds', type=int, default=5, help='number of folds')
    parser.add_argument('-g', '--gpu_id', type=str, default='0', help='gpuid used for trianing')
    parser.add_argument('-m', '--model', type=str, default='plain', help='which model to be used')
    parser.add_argument('--img_size', type=int, default=1088, help='number of folds')
    parser.add_argument('--save_train', type=bool, default=False, help='save the reconstruction results for training data')

    args = parser.parse_args()

    if not os.path.exists(os.path.join('results/', args.exp_name, 'models')):
        os.mkdir(os.path.join('results/', args.exp_name))
        os.mkdir(os.path.join('results/', args.exp_name, 'models'))
        os.mkdir(os.path.join('results/', args.exp_name, 'reconstruction_train'))
        os.mkdir(os.path.join('results/', args.exp_name, 'reconstruction_test'))

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    data_path = '/disk/songwei/LockheedMartion/end2end/MVS/'
    kml_path = '/disk/songwei/LockheedMartion/end2end/KML/'
    gt_path = '/disk/songwei/LockheedMartion/end2end/DSM/'
    data_file = './results/data_small.npz'
    train_dataset= data_util.MVSdataset_lithium(data_file)
    # train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    # import ipdb;ipdb.set_trace()
    trainer = Trainer(args, train_dataset)
    if args.load_model:
        trainer.D.load_state_dict(torch.load(os.path.join('../results', args.exp_name, 'models', '%s_%d'%(args.input_fold, args.input_epoch))))
    trainer.run()
