import matplotlib.pyplot as plt
import numpy as np

def plot_dataset(x, y, legend_loc='lower right'):
    fig, ax = plt.subplots()
    ax.scatter(x[y==1, 0], x[y==1, 1], c='r', s=100, alpha=0.7, marker='*', label='Salmon',linewidth=0)
    ax.scatter(x[y==- 1, 0], x[y==-1, 1], c='b', s=50, alpha=0.7, marker='o', label='Sea Bass',linewidth=0)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.set_xlabel('Length')
    ax.set_ylabel('Lightness')
    ax.set_aspect('equal')
    if legend_loc:
        ax.legend(loc=legend_loc,fancybox=True).get_frame().set_alpha(0.5)
        ax.grid('on')

X = np.array([[2, 1], [0, -1], [1.5, 0], [0, 1], [-1, 1], [-3, 0],[1, -1], [2, - 1], [3, -2], [3, 1], [-2, 1.5], [-3, 0.5], [-1, 2]])
Y = np.array([1, 1, 1, -1, -1, -1,1, 1, 1, 1, -1, - 1, -1])
#X = np.vstack([X, np.array([0, -0.2])])
#Y = np.hstack([Y, np.array([-1])])
plot_dataset(X, Y)
#order cat y
SB = [(x[0], x[1]) for x,y in zip(X,Y) if y ==1]
SN = [(x[0], x[1]) for x,y in zip(X,Y) if y ==-1]
dif = np.array([[((xb[0]-xn[0])**2 + (xb[1]-xn[1])**2)**0.5 for xb in SB] for xn in SN])
difm = dif.min()
difmp = np.argmin(dif)

# Add dummmy vector
ones = np.array([[float(1) for i in range(X.shape[1]+1)] for i in range(len(X))])
ones[:,1:] = X
X = ones
# Define initial values
t = 0
b = difm/2
w = np.array([0,0,0])
gamma = 1
Z = 100
while Z >= 1/gamma**2 and t < 100:
    nhe = .5
    #nw = sum([s**2 for s in w])**0.5
    for x,y in zip(X,Y):    
        if y == np.sign(np.dot(w,x)-b-0.0001):
            w = w
        elif y <= np.sign(np.dot(w,x)-b-0.0001):
            w = w - x*nhe
        else:
            w = w + x*nhe
    Z = sum([abs(y-(np.sign(np.dot(w,x)-b-0.0001))) for x,y in zip(X,Y)])
    t = t+1
xcl = range(-6,6,1)
classif = [(w[0]-b) + (-w[1]/w[2])*i for i in xcl]
plt.ylim(-6,6)
plt.yticks(np.arange(-6, 6, step=1))
plt.xlim(-6,6)
plt.xticks(np.arange(-6, 6, step=1))
plt.plot(xcl, classif)
plt.savefig("linear classifier1.png")
plt.show()
print(round((w[0]-b),2),'is the intercept of the linear classifier and',round(-w[1]/w[2],2),'is its slope.')
print(t)