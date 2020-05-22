import os
import matplotlib.pyplot as plt

# model1 = 'mvsnet_large0'
model2 = 'mvsnet_large99'
model3 = 'mvsnet_large239'
models = [model2, model3]
# names = ['no_pretrain', '50 epochs', '250 epochs']
names = ['50 epochs', '250 epochs']

metrics = {'l2':[], 'l1':[], 'completeness':[], 'median error':[]}
for modelname in models:
    start_epoch = int(modelname[12:])
    metrics['l2'].append([])
    metrics['l1'].append([])
    metrics['completeness'].append([])
    metrics['median error'].append([])
    with open(os.path.join('debug/end2end_mvsnet_large_pretrain%d_1000/log_test_bak.txt'%start_epoch)) as f:
        for line in f:
            metrics['l2'][-1].append(float(line.rstrip().split()[-18]))
            metrics['l1'][-1].append(float(line.rstrip().split()[-3]))
            metrics['completeness'][-1].append(float(line.rstrip().split()[-8]))
            metrics['median error'][-1].append(float(line.rstrip().split()[-13]))

colors = ['orange', 'turquoise', 'firebrick']
# plt.style.use('bmh')
min_epochs = min([len(loss) for loss in metrics['l2']])
n_epoch = [i+1 for i in range(min_epochs)]

for j, metric_name in enumerate(metrics.keys()):
    ax1 = plt.subplot(2, 2, j+1)
    for i in range(len(metrics[metric_name])):
        ax1.plot(n_epoch, metrics[metric_name][i][:min_epochs], colors[i], label=names[i])
    ax1.set_xlabel('number of epochs')
    ax1.set_ylabel(metric_name)
    ax1.grid(True)
    ax1.legend()

plt.savefig('debug/test_metrics.png')