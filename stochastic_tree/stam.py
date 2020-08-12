import torch
import model_conf
import lib
import matplotlib.pyplot as plt
from params import parameters

prms = parameters()


PATH = './model'
device = torch.device("cuda")
net = model_conf.Forest(prms)
net.load_state_dict(torch.load(PATH))
net.to(device)

X,Y = lib.generate_circle()

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