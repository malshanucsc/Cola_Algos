#coding:utf-8
#Passive-aggressive
#Binary Classification Algorithm

import sys
import numpy as np
from pylab import *

def PA(C, loss, xn):
    return loss/xn

def PA1(C, loss, xn):
    return min([C, loss/xn])

def PA2(C, loss, xn):
    return loss/(xn+1/(2*C))

def PA_single(X, label, C, num_loop):
    D = len(X[0,:])
    w = np.zeros(D+1)
    res = []
    for loop in range(num_loop):
        for idx, x in enumerate(X):
            x = concatenate((x,[1]))
            yh = label[idx]*np.dot(w, x)
            if yh<1:
                loss = 1-yh
                xn = np.sum(x**2)
                tau = PA1(C, loss, xn)
                w = w+label[idx]*tau*x
        if 0<loop:
            res.append(sum((w-wd)**2))
        wd = np.copy(w)
    return w, res

if __name__ == '__main__':

    N = 400
    c1 = [2, 2]
    c2 = [2, -2]
    cv = [[0.5, 0.3], [0.3, 0.5]]
    d1 = np.random.multivariate_normal(c1, cv, N/2)
    d2 = np.random.multivariate_normal(c2, cv, N/2)
    D = vstack((d1, d2))

    label = zeros(len(D))
    for k in range(N):
        if k<N/2:
            label[k] = 1
        else:
            label[k] = -1

    num_loop = 100
    C = 1

    w, res = PA_single(D, label, C, num_loop)

    labeld = []
    for x in D:
        x = concatenate((x,[1]))
        labeld.append(np.sign(np.dot(w,x)))

    ion()
    figure(1)
    clf()
    chk = where(label!=labeld)[0]
    for n in chk:
        scatter(D[n,0], D[n,1], s=80, c='g', marker='o')

    x = d1[:,0]
    y = d1[:,1]
    scatter(x, y, c='b', marker='o')

    x = d2[:,0]
    y = d2[:,1]
    scatter(x, y, c='r', marker='o')

    figure(2)
    clf()
    plot(res)

