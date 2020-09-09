import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    margin = 0.1
    x_min, x_max = x.min() - margin, x.max() + margin
    y_min, y_max = y.min() - margin, y.max() + margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def generate_circle(samples=1000,dim=2):
    X = np.random.rand(samples,dim)
    Y = np.zeros(samples)
    Y[(X[:,0]-0.5)**2+(X[:,1]-0.5)**2<0.16]=1

    return X,Y

def generate_triangle(samples=1000,dim=2):
    X = np.random.rand(samples,dim)
    Y1 = np.zeros(samples)
    Y2 = np.zeros(samples)
    Y1[X[:,0]<X[:,1]]=1
    Y2[X[:,0]>1-X[:,1]]=1
    Y = Y1*Y2
    return X,Y

def generate_angle(samples=1000,dim=2):
    X = np.random.rand(samples,dim)
    Y1 = np.zeros(samples)
    Y2 = np.zeros(samples)
    Y1[X[:,0]<X[:,1]]=1
    # Y2[X[:,0]>1-X[:,1]]=1
    Y = Y1#*Y2
    return X,Y

def plot_2d_function(ax, w, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    w: weights+bias
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = w[0]*xx.ravel()+w[1]*yy.ravel()+w[2]
    Z = 1/(1 + np.exp(-Z))
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out