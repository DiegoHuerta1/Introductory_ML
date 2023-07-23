import numpy as np
import matplotlib.pyplot as plt
from math import e

def sigmoid(x):
    if type(x) != 'numpy.ndarray':
        return 1 / (1+e **(-x))
    elif len(x.shape) == 1 or x.shape[1] == 1:
        m = np.ones(len(x))
        for i in range(len(x)):
            m[i] = 1 / (1+e **(-x[i]))
        return m
    else:
        x2 = x.reshape(x.size)
        m2 = sigmoid(x2)
        m = m2.reshape(x.shape)
        return m

def sigmoid_dif(x):
    if type(x) != 'numpy.ndarray':
        return sigmoid(x)*(1-sigmoid(x))
    elif len(x.shape) == 1 or x.shape[1] == 1:
        m = np.ones(len(x))
        for i in range(len(x)):
            m[i] = 1 / (1+e **(-x[i])) * (1-1 / (1+e **(-x[i])))
        return m
    else:
        x2 = x.reshape(x.size)
        m2 = sigmoid_dif(x2)
        m = m2.reshape(x.shape)
        return m

class Neural_Net:
    def __init__(self, size_list):
        self.depth = len(size_list) - 1
        self.nodes = np.array(size_list)
        self.nodes[0] += 1
        self.size = self.nodes.sum()
        self.width = self.nodes.max()
        number_w = 0
        for i in range(self.depth):
            number_w += self.nodes[i] * self.nodes[i+1]
        self.weigths = (np.random.random(number_w) - 0.5)

    def print_info(self):
        print('Depth of the neural net: '+str(self.depth))
        print('Nodes : ' + str(self.nodes))
        print('Size: '+str(self.size))
        print('Width: ' + str(self.width))
        print('Number of parameters: ' + str(len(self.weigths)))
        print('All weigths: ')
        print(self.weigths)

    def loss(self, x, y):
        predict = self.predict(x)
        if len(predict) == 1:
            return 1/2*(predict-y)**2
        else:
            return np.dot(predict - y, predict - y)*1/2

    def train(self, X, Y, iterations=1000, lam=0, reduct_step = 3 , print_iter = 1):
        step_size = reduct_step
        hist = np.zeros([3, iterations])
        maximum = -10
        for t in range(iterations):
            i = np.random.randint(0, X.shape[0])
            x = X[i]
            y = Y[i]
            v = self.backpropagation(x, y)
            if t != 0:
                step_size = reduct_step /(t**(1/2))
            self.weigths = self.weigths - step_size*(v + lam*self.weigths)
            #test
            predictions_m = self.predict(X)
            loss = self.loss(X, Y)
            predictions_m[predictions_m <= 0.5] = 0
            predictions_m[predictions_m > 0.5] = 1
            pre_m = sum(predictions_m == Y) / X.shape[0]
            if pre_m >= maximum:
                maximum = pre_m
                weigths = self.weigths.copy()
            hist[0, t] = pre_m
            hist[1, t] = loss
            if print_iter == 1:
                print('Iteration: '+str(t+1)+' : '+str(pre_m)+'  '+str(loss))
        print('Trained')
        self.weigths = weigths
        return hist

    def backpropagation(self, x, y):
        output = self.predict(x)
        gradient = np.array([])
        Delta = np.zeros([len(self.nodes), self.width])
        delta_final = output - y
        for t in np.arange(Delta.shape[0]-1, 0, -1):
            for i in range(self.nodes[t]):
                if t == self.depth:
                    Delta[t, i] = delta_final[i]
                else:
                    W = self.weigth_matrix(t)
                    d = 0
                    for j in range(self.nodes[t+1]):
                        d += W[j, i] * Delta[t+1,j]*sigmoid_dif(self.activation(t+1, j, x))
                    Delta[t, i] = d

        for t in range(self.depth):
            W = self.weigth_matrix(t)
            G = np.zeros(W.shape)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    G[i, j] = Delta[t+1, i] * sigmoid_dif(self.activation(t+1, i, x))*self.output(t, j, x)
            gradient = np.concatenate([gradient,G.reshape(G.size)])
        return gradient

    def activation(self, t, indx, x):
        # activation of node indx in layer t, with input vector x
        if x.ndim == 1 and self.nodes[0] > 2:
            activations = np.concatenate([x, np.ones(1)])
        elif x.ndim == 0 and self.nodes[0] == 2:
            activations = np.array([x, 1])
        else:
            ones = np.ones(x.shape[0])
            activations = (np.column_stack([x,ones.T])).T
        for i in range(t):
            W = self.weigth_matrix(i)
            activations = np.dot(W, activations)
            if i != t-1:
                activations = sigmoid(activations)
        return activations[indx]

    def output(self, t, indx, x):
        # output of node indx in layer t, with input vector x
        if x.ndim == 1 and self.nodes[0] > 2:
            activations = np.concatenate([x, np.ones(1)])
        elif x.ndim == 0 and self.nodes[0] == 2:
            activations = np.array([x, 1])
        else:
            ones = np.ones(x.shape[0])
            activations = (np.column_stack([x,ones.T])).T
        for i in range(t):
            W = self.weigth_matrix(i)
            activations = np.dot(W, activations)
            activations = sigmoid(activations)
        return activations[indx]


    def weigth_matrix(self, t):
        #devuelve la matriz de pesos de la capa t, a la capa t+1
        before = 0
        for i in range(t):
            before += self.nodes[i] * self.nodes[i+1]
        w = self.weigths[before: before + self.nodes[t]*self.nodes[t+1]]
        W = w.reshape([self.nodes[t+1], self.nodes[t]])
        if t != self.depth -1:
            W[-1] = 0
        return W

    def predict(self, X):
        if X.ndim == 1 and self.nodes[0] > 2:
            activations = np.concatenate([X, np.ones(1)])
        elif X.ndim == 0 and self.nodes[0] == 2:
            activations = np.array([X, 1])
        else:
            ones = np.ones(X.shape[0])
            activations = (np.column_stack([X,ones.T])).T
        for i in range(self.depth):
            W = self.weigth_matrix(i)
            activations = np.dot(W, activations)
            activations = sigmoid(activations)
        if self.nodes[self.depth] == 1:
            return (activations.T).flatten()
        else:
            return activations.T
