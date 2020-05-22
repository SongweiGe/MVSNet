import os
import matplotlib.pyplot as plt

train_losses = []
metrics = {'l2':[], 'l1':[], 'completeness':[], 'median error':[]}

modelname = 'mvsnet_large239'
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
with open(os.path.join('logs', modelname+'.txt')) as f:
    for line in f:
        if 'average' not in line:
            continue
        train_losses.append(float(line.rstrip().split()[-1]))

plt.style.use('bmh')
train_epochs = len(train_losses)
min_epochs = min([len(loss) for loss in metrics['l2']])
n_train_epoch = [i+1 for i in range(train_epochs)]
n_epoch = [i*5+1 for i in range(min_epochs)]
fig, ax1 = plt.subplots(1, 1)
ax1.plot(n_train_epoch, train_losses, 'black', label='train')
ax1.plot(n_epoch, metrics['l1'][0], linestyle='--', color='black', label='validation')
ax1.set_xlabel('number of epochs')
ax1.set_ylabel('loss')
ax1.legend()
ax1.grid(True)
plt.savefig('debug/all_losses.png')