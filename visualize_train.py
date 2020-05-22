import os
import matplotlib.pyplot as plt

model1 = 'mvsnet_large0'
model2 = 'mvsnet_large99'
model3 = 'mvsnet_large239'
# models = [model1, model2, model3]
models = [model1, model3]
# names = ['no_pretrain', '50 epochs', '250 epochs']
names = ['no_pretrain', '250 epochs']

losses = []
for modelname in models:
    start_epoch = int(modelname[12:])
    losses.append([])
    with open(os.path.join('logs', modelname+'.txt')) as f:
        for line in f:
            if 'average' not in line:
                continue
            losses[-1].append(float(line.rstrip().split()[-1]))

colors = ['orange', 'turquoise', 'firebrick']
plt.style.use('bmh')
min_epochs = min([len(loss) for loss in losses])
n_epoch = [i+1 for i in range(min_epochs)]
fig, ax1 = plt.subplots(1, 1)
for i in range(len(losses)):
    ax1.plot(n_epoch, losses[i][:min_epochs], colors[i], label=names[i])
ax1.set_xlabel('number of epochs')
ax1.set_ylabel('loss on training dataset')
ax1.grid(True)
ax1.legend()
plt.savefig('debug/pretrain_losses.png')