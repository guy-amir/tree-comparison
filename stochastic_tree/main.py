#load libraries
from params import parameters
from dl import get_dataloaders
from model_conf import Forest
from train_conf import Trainer
import pandas as pd
import lib
import matplotlib.pyplot as plt

#load default parameters (including device)
prms = parameters()

#dataloaders
trainset, testset, trainloader, testloader = get_dataloaders(prms)

def evaluate_network(prms):
    #dataloaders
    # trainset, testset, trainloader, testloader = get_dataloaders(prms)

    ###here we will add a loop that will hange dataset size

    #initiate model:
    net = Forest(prms)
    net.to(prms.device) #move model to CUDA

    #run\fit\whatever
    trainer = Trainer(prms,net)
    loss_list,val_acc_list,train_acc_list,wav_acc_list,cutoff_list,smooth_list = trainer.fit(trainloader,testloader)
    
    return net

net = evaluate_network(prms)

X,Y = lib.generate_circle(samples = 2000)

fig, sub = plt.subplots(1,1)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

ax = sub

X0, X1 = X[:, 0], X[:, 1]
xx, yy = lib.make_meshgrid(X0, X1)

# for clf, title, ax in zip(models, titles, sub.flatten()):
lib.plot_2d_function(ax, net, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('X0')
ax.set_ylabel('X1')
ax.set_xticks(())
ax.set_yticks(())
# ax.set_title(title)

plt.show()
plt.savefig('./figs/circle_td2_e400_acc78.5_small_pi.png')