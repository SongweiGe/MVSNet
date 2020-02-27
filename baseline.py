import os
import numpy as np
import imageio as misc
from shutil import copyfile

from utils import eval_util

def partial_median(x):
    k = x.shape[0]//2
    med = np.median(x[:k][x[k:].astype(bool)])
    return med

n_pair = 6
save_fig = True
# mode = 'test_outdomain'
mode = 'test'
input_temp = 'FF-%d.npy'
input_img_temp = '%d-color.png'
result_path = './results/baseline'
data_path = '/disk/songwei/LockheedMartion/end2end/'
input_path = '/disk/songwei/LockheedMartion/end2end/MVS'
gt_path = '/disk/songwei/LockheedMartion/end2end/DSM'

if not os.path.exists(result_path):
    os.mkdir(result_path)

fire_palette = misc.imread(os.path.join('image/fire_palette.png'))[0][:, 0:3]

asp_scores = []

filenames = [line.rstrip() for line in open('data/file_lists.txt') if int(line.rstrip().split('_')[2]) <= 33]
print('Number of test data: %d'%len(filenames))
for filename in filenames:
    out_path = os.path.join(result_path, filename)
    print(filename+':')
    pair_data = []

    height = np.load(os.path.join(input_path, filename, input_temp%0))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(height, fire_palette)
        misc.imsave(out_path+'_input_height%d.png'%0, color_map)
        copyfile(os.path.join(input_path, filename, input_img_temp%0), out_path+'_input_color%d.png'%0)

    # load gt
    gt_data = np.load(os.path.join(gt_path, filename+'.npy'))
    if save_fig:
        color_map = eval_util.getColorMapFromPalette(gt_data, fire_palette)
        misc.imsave(out_path+'_gt_data.png', color_map)

    metrics = eval_util.evaluate(height, gt_data)
    asp_scores.append(metrics)
    print('RMSE, Accuracy, Completeness, L1 Error of consense vote: %.4f,  %.4f,  %.4f,  %.4f'%(
            metrics[0], metrics[1], metrics[2], metrics[3]))

asp_scores = np.squeeze(np.array(asp_scores))

print('Final RMSE, Accuracy, Completeness, L1 Error of weighted deep vote: mean = %.4f,  %.4f,  %.4f,  %.4f, std = %.4f,  %.4f,  %.4f,  %.4f'%
    tuple(np.mean(asp_scores, 0).tolist()+np.std(asp_scores, 0).tolist()))