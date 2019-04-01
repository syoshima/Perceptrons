# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:48:12 2019

@author: Samy Abud Yoshima
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_dataset(x, y, legend_loc='upper right'):
    fig, ax = plt.subplots()
    ax.scatter(x[y==1, 0], x[y==1, 1], c='r', s=100, alpha=0.7, marker='*', label='Sea Bass',linewidth=0)
    ax.scatter(x[y==- 1, 0], x[y==-1, 1], c='b', s=100, alpha=0.7, marker='o', label='Salmon',linewidth=0)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_xlabel('Length')
    ax.set_ylabel('Lightness')
    ax.set_aspect('equal')
    if legend_loc:
        ax.legend(loc=legend_loc,fancybox=True).get_frame().set_alpha(0.5)
        ax.grid('on')
# Data
X = np.array([[4, 1], [-2, 1], [ -2, - 4], [-1, -1], [2, -1], [-1, -3], [3, 2], [1, 2.5], [-3, -1], [-3, 3], [0,-2], [4, -2], [3, -4]])
Y = np.array([1, 1, 1, -1, - 1, -1, 1, 1, 1, 1, -1, -1, -1])
plot_dataset(X, Y)
#order cat y
SB = [(x[0], x[1]) for x,y in zip(X,Y) if y ==1]
SN = [(x[0], x[1]) for x,y in zip(X,Y) if y ==-1]
dif = np.array([[((xb[0]-xn[0])**2 + (xb[1]-xn[1])**2)**0.5 for xb in SB] for xn in SN])
difm = dif.min()
# Kernel Trick
X = np.array([(1/(1+np.exp(x1)),1/(1+np.exp(-x2))) for x1,x2 in zip(X[:,0],X[:,1])])
# Add dummmy vector
ones = np.array([[float(1) for i in range(X.shape[1]+1)] for i in range(len(X))])
ones[:,1:] = X
X = ones
# Define initial values
t = 0
b = difm
w = np.array([0,0,0])
gamma = 1
Z = 100
while Z >= 1/gamma**2 and t < 10000:
    nw = sum([s**2 for s in w])**0.5
    nhe = 0.5
    for x,y in zip(X,Y):    
        if y == np.sign(np.dot(w,x)-b-0.0001):
            w = w
        elif y <= np.sign(np.dot(w,x)-b-0.0001):
            w = w - x*nhe
        else:
            w = w + x*nhe
    Z1 = ([(((np.dot(w,x)-b-0.0001))) for x in (X)])
    Z2 = ([((np.sign(np.dot(w,x)-b-0.0001))) for x in (X)])
    Z3 = ([abs(y-(np.sign(np.dot(w,x)-b-0.0001))) for x,y in zip(X,Y)])
    Z = sum([abs(y-(np.sign(np.dot(w,x)-b-0.0001))) for x,y in zip(X,Y)])
    t=t+1
# plot solution
xcl = np.arange(-6,6,0.5)
classif = [[(w[0]-b)+w[1]*1/(1+np.exp(i))+w[2]*1/(1+np.exp(-j)) for j in xcl] for i in xcl]
plt.ylim(-6,6)
plt.yticks(np.arange(-6, 6, step=0.5))
plt.xlim(-6,6)
plt.xticks(np.arange(-6, 6, step=0.5))
plt.plot(xcl, classif)
plt.savefig("nonlinear classifier_sigmoid.png")
plt.show()
print(t)
print(w)
xx, yy = np.meshgrid(xcl, xcl, sparse=True)
z = (w[0]-b)+w[1]*1/(1+np.exp(xx))+w[2]*1/(1+np.exp(-yy))
h = plt.contourf(xcl,xcl,z)
plt.show()
'''
while Z >= 1/gamma**2 and t < 1040:
    nw = sum([s**2 for s in w])**0.5
    nhe = 1
    for x,y in zip(X,Y):    
        if   y == np.sign(-w[0]*x[0]+1/(1+np.exp(w[1]*x[1]))+1/(1+np.exp(w[2]*-x[2]))):
            w = w
        elif y <= np.sign(-w[0]*x[0]+1/(1+np.exp(w[1]*x[1]))+1/(1+np.exp(w[2]*-x[2]))):
            w = w - x*nhe
        else:
            w = w + x*nhe
    Z1 =     [(-w[0]*x[0]+1/(1+np.exp(w[1]*x[1]))+1/(1+np.exp(w[2]*-x[2]))) for x in X]
    Z2 =     [np.sign(-w[0]*x[0]+1/(1+np.exp(w[1]*x[1]))+1/(1+np.exp(w[2]*-x[2]))) for x in X]
    Z3 =     [abs(y-(np.sign(-w[0]*x[0]+1/(1+np.exp(w[1]*x[1]))+1/(1+np.exp(w[2]*-x[2]))))) for x,y in zip(X,Y)]
    Z  = sum([abs(y-(np.sign(-w[0]*x[0]+1/(1+np.exp(w[1]*x[1]))+1/(1+np.exp(w[2]*-x[2]))))) for x,y in zip(X,Y)])
    t = t+1


'''