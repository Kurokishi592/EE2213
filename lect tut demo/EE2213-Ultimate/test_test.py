import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def ReLU(x):
    return x*(x>0)

def test(x):
    return(x+1)

X = np.array([
    [1, 2, 3]
])
Y = np.array([
    [0.1,0.9]
])
W1 = np.array([
    [0.1, 0.1],
    [-0.1, 0.2],
    [0.3, -0.4]
])
W2 = np.array([
    [0.1, 0.1],
    [0.5, -0.6],
    [0.7, -0.8]
])

f = np.vectorize(ReLU)
layer1=f(X@W1)
print('f(X@W1)',layer1)
inner2=np.hstack((np.ones((len(layer1),1)),layer1))
layer2=f(inner2@W2)
print('f(inner2)',layer2)
# inner3=np.hstack((np.ones((len(layer2),1)),layer2))
# layer3=f(inner3@W2)
# print('layer3', layer3)






