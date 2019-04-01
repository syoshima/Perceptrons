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
X = np.array([(max(-x1,0),max(x2,0)) for x1,x2 in zip(X[:,0],X[:,1])])
# Add dummmy vector
ones = np.array([[float(1) for i in range(X.shape[1]+1)] for i in range(len(X))])
ones[:,1:] = X
X = ones
# Define initial values
t = 0
b = difm/2
w = np.array([10,-10,10])
gamma = 1
Z = 100
while Z >= 1/gamma**2 and t < 1000:
    nw = sum([s**2 for s in w])**0.5
    nhe = .2
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
# x1>0, x2>0
#clpp = [-theta/w[2] for i in x1cl if i >= 0]
# x1<0, x2>0
#clnp = [-theta/w[2] + -w[1]/w[2]*i for i in x1cl if i <= 0 and i>-theta/w[1]]
# x1<0, x2<0
#clnn = [-theta/w[1] for any x2]
#clpn = []
theta = w[0]-b
x1cl = np.arange(-6-abs(theta/w[1]),6+abs(theta/w[1]),abs(theta/w[1]))
classif = []
for i in x1cl:
    if i>=0:
        c = -theta/w[2]
    elif i < 0 and i >=-abs(theta/w[1]):
        c = -theta/w[1]
    else:
        c = -100
    classif.append(c)   
# x1>0, x2<0 : theta = 0 --> w[0] = b ???? or that w[0] = 0
plt.ylim(int(-6-abs(theta/w[1])),int(6+abs(theta/w[1])))
plt.yticks(np.arange(int(-6-abs(theta/w[1])),int(6+abs(theta/w[1])), step=abs(theta/w[1])))
plt.xlim(int(-6-abs(theta/w[1])),int(6+abs(theta/w[1])))
plt.xticks(np.arange(int(-6-abs(theta/w[1])),int(6+abs(theta/w[1])), step=abs(theta/w[1])))
plt.plot(x1cl, classif)
plt.savefig("nonlinear classifier_relu.png")
plt.show()
print(t)
print(w,b)
