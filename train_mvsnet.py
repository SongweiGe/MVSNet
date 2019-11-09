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

from triangulationRPC_matrix_torch import triangulationRPC_matrix

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, args, dataloader):
        self.args = args
        self.n_folds = args.n_folds
        self.n_epochs = args.epochs
        self.batch_size = args.batch_size
        self.D = FlowNetS().cuda()
        # self.L = nn.MSELoss().cuda()
        self.L = nn.L1Loss().cuda()
        self.optimizer = torch.optim.Adadelta(self.D.parameters(), lr=1.)
        self.dataloader = dataloader
        self.n_total = len(self.dataloader)
        self.shuffled_index = np.arange(self.n_total)
        np.random.seed(2019)
        np.random.shuffle(self.shuffled_index)
        self.wgs84 = pyproj.Proj('+proj=utm +zone=21 +datum=WGS84 +south')


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

    def triangulation_forward(self, disparity_map, nrow, ncol, rpc_pair, h_pair, area_info):
        rpc_l, rpc_r = rpc_pair
        h_left_inv, h_right_inv = h_pair
        bbox, bounds, im_size = area_info
        # import ipdb;ipdb.set_trace()
        cu1, ru1 = torch.meshgrid([torch.arange(ncol), torch.arange(nrow)])
        cu1 = cu1.type(torch.cuda.DoubleTensor).transpose(1, 0)
        ru1 = ru1.type(torch.cuda.DoubleTensor).transpose(1, 0)
        ru2 = ru1 + disparity_map[0, 0, :, :].type(torch.cuda.DoubleTensor)
        cu2 = cu1

        cu1, ru1 = self.apply_homography(h_left_inv, [cu1, ru1])
        cu2, ru2 = self.apply_homography(h_right_inv, [cu2, ru2])
        ru1 = ru1.reshape(-1)
        cu1 = cu1.reshape(-1)
        ru2 = ru2.reshape(-1)
        cu2 = cu2.reshape(-1)
        Xu, Yu, Zu, _, _ = triangulationRPC_matrix(ru1, cu1, ru2, cu2, rpc_l, rpc_r, verbose=False)

        xx, yy = np.meshgrid(np.arange(im_size[1]), np.arange(im_size[0]))
        lons, lats = self.wgs84(Xu, Yu)
        ix, iy = geo_utils.spherical_to_image_positions(lons, lats, bounds, im_size)

        int_im = griddata((iy, ix), Zu, (yy, xx))
        int_im = geo_utils.fill_holes(int_im)
        return int_im[bbox[0]:bbox[1], bbox[2]:bbox[3]]

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
            test_ids = self.shuffled_index[list(range(i*fold_size, (i+1)*fold_size))]
            train_ids = self.shuffled_index
            # train_ids = np.setdiff1d(self.shuffled_index, test_ids)
            # import ipdb;ipdb.set_trace()
            n_batch = train_ids.shape[0]//self.batch_size
            n_test_batch = test_ids.shape[0]//self.batch_size
            # for p in self.D.parameters():
            #     self.weights_init(p)
            for j in range(self.n_epochs):
                np.random.shuffle(train_ids)
                begin = time.time()
                train_epoch_loss = []
                test_epoch_loss = []
                # import ipdb;ipdb.set_trace()
                for k in range(n_batch+1):
                    #forward calculation and back propagation, X: B x P x 2 x W x H
                    # import ipdb;ipdb.set_trace()
                    if k == n_batch:
                        train_batch_ids = train_ids[k*self.batch_size:]
                    else:
                        train_batch_ids = train_ids[k*self.batch_size:(k+1)*self.batch_size]
                    img_pair, h_pair, rpc_pair, area_info, y = self.dataloader.__getitem__(train_batch_ids[0])
                    X = Variable(torch.cuda.FloatTensor([img_pair]), requires_grad=False)
                    Y = Variable(torch.cuda.FloatTensor([y]), requires_grad=False)
                    h_pair = torch.cuda.DoubleTensor(np.stack(h_pair))
                    self.optimizer.zero_grad()
                    disparity_map = self.D(X)[0]
                    # import ipdb;ipdb.set_trace()
                    pred_height = self.triangulation_forward(disparity_map, X.shape[2], X.shape[3], rpc_pair, h_pair, area_info)
                    loss = self.L(pred_height, Y)
                    loss_val = loss.data.cpu().numpy()
                    loss.backward()
                    self.optimizer.step()
                    train_epoch_loss.append(loss_val)
                    del X,Y,pred_height,loss
                
                if (j+1)%100 == 0:
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
                img_pair, h_pair, rpc_pair, area_info, y = self.dataloader.__getitem__(train_batch_ids[0])
                X = Variable(torch.cuda.FloatTensor(img_pair), requires_grad=False)
                Y = Variable(torch.cuda.FloatTensor(y), requires_grad=False)
                diaparity_map = self.D(X)[0]
                pred_height = self.triangulation_forward(disparity_map, X.shape[1], X.shape[2], rpc_pair, h_pair, area_info)
                loss = self.L(pred_height, Y)
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
    parser.add_argument('--save_train', type=bool, default=False, help='save the reconstruction results for training data')

    args = parser.parse_args()

    if not os.path.exists(os.path.join('results/', args.exp_name, 'models')):
        os.mkdir(os.path.join('results/', args.exp_name))
        os.mkdir(os.path.join('results/', args.exp_name, 'models'))
        os.mkdir(os.path.join('results/', args.exp_name, 'reconstruction_train'))
        os.mkdir(os.path.join('results/', args.exp_name, 'reconstruction_test'))

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    data_path = '/disk/songwei/LockheedMartion/end2end/MVS/'
    kml_path = '/disk/songwei/LockheedMartion/DeepVote/kml/'
    gt_path = '/disk/songwei/LockheedMartion/DeepVote/DSM/'
    train_dataset= data_util.MVSdataset(gt_path, data_path, kml_path)
    # train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    trainer = Trainer(args, train_dataset)
    if args.load_model:
        trainer.D.load_state_dict(torch.load(os.path.join('../results', args.exp_name, 'models', '%s_%d'%(args.input_fold, args.input_epoch))))
    trainer.run()