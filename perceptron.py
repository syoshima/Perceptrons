# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:26:39 2019

@author: Samy Abud Yoshima
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_dataset(x, y, legend_loc='lower right'):
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

        # For the three figures in part 1
x = np.array([[2, 1], [0, -1], [1.5, 0], [0, 1], [-1, 1], [-3, 0],[1, -1], [2, - 1], [3, -2], [3, 1], [-2, 1.5], [-3, 0.5], [-1, 2]])
y = np.array([1, 1, 1, -1, -1, -1,1, 1, 1, 1, -1, - 1, -1])
plot_dataset(x, y)
xcl = range(-5,5,1)
classif = [i for i in xcl]
plt.plot(xcl, classif)

x2 = np.vstack([x, np.array([0, -0.2])])
y2 = np.hstack([y, np.array([-1])])
plot_dataset(x2, y2)
xcl2 = range(-5,5,1)
classif2 = [-0.7+i for i in xcl2]
plt.plot(xcl, classif2)

x3 = np.array([[4, 1], [-2, 1], [ -2, - 4], [-1, -1], [2, -1], [-1, -3], [3, 2], [1, 2.5], [-3, -1], [-3, 3], [0,-2], [4, -2], [3, -4]])
y3 = np.array([1, 1, 1, -1, - 1, -1, 1, 1, 1, 1, -1, -1, -1])
plot_dataset(x3, y3, legend_loc='lower right')
xcl3 = np.arange(-5,5,step=0.5)
classif31 = []
for i in xcl3:
   if i < -1.5:
        c3 = abs((i)**len(x3))
        
   else:
        c3 = 0
   classif31.append(c3)
classif3 = [(-(j)/(1+np.exp(-i))) for i,j in zip(xcl3,classif31)]
#yticks(np.arange(-5, 5, step=0.1))
plt.ylim(-5,5)
plt.yticks(np.arange(-5, 5, step=0.5))
plt.xlim(-5,5)
plt.xticks(np.arange(-5, 5, step=0.5))
plt.plot(xcl3, classif3,label="Binary Classifier")
plt.legend(loc='upper left')



        # For the sigmoid network in part 2
def sigmoid(inputs):
    return 1.0 / (1.0 + np.exp(-inputs))

def nn_2layer(inputs):
    return np.sign(sigmoid(inputs[:, 0]) + sigmoid(-inputs[:, 1]) - 1.5)

def plot_decision_boundary(network):
    x0v, x1v = np.meshgrid(np.linspace(-2, 8, 20), np.linspace(-8, 2, 20))
    x4 = np.hstack([x0v.reshape((-1,1)), x1v.reshape((-1,1))])
    #print(x4)
    y4 = network(x4)
    #print(y4)
    plot_dataset(x4, y4, legend_loc=None)

plot_decision_boundary(nn_2layer)
# For the ReLU network in Part 2
def relu(inputs):
    return np.maximum(0, inputs)
    
def nn_2layer_relu(inputs):
    return np.sign(relu(-inputs[:, 0]) + relu(inputs[:, 1]) - 0.1)
plot_decision_boundary(nn_2layer_relu)    

