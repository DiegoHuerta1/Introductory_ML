import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import funciones # para el mapeo

# no importa si se quiere trabajar en dimensiones d muy alto
# los svm controlan el sample complexity
# los kernels controlan la dificultad computacional

puntos = 100
X = np.random.random((puntos, 2))*4-2  # -2 a 2
y = np.ones(puntos)*-1

p1 = np.ones(puntos) # predict 1
p2 = np.zeros(puntos) #predict 2
pre1 = 0 # precicion
pre2 = 0 #precicion 2

for i in range(puntos):
    if 1.5 <= X[i, 0]*X[i, 0] + X[i, 1]*X[i, 1] <= 2: # margen
        X[i, 0] = np.random.random()
        X[i, 1] = np.random.random()
        y[i] = 1
    elif X[i, 0]*X[i, 0] + X[i, 1]*X[i, 1] < 1.5: # dentro
        y[i] = 1

M = funciones.circ(X) # mapeo

w1 = np.zeros(len(M[0]))
w2 = np.zeros(len(M[0]))

# preceptron
error = 0
for j in range(0, 100):
    for i in range(0, puntos):
        if np.dot(w1, M[i, :]) * y[i] <= 0:
            w1 = w1 + y[i] * M[i, :]
            error = 1
    if error == 0:
            break

#svm with kernels
T = 1000

beta = np.zeros(puntos)
alpha = np.zeros([T, puntos])

for t in range(0, T):
    alpha[t, :] = beta
    i = np.random.randint(0, puntos)
    p = 0
    for j in range(0, puntos):
        xi = X[i, :]
        xj = X[j, :]
        p = p + alpha[t, j] * funciones.kp2(xi, xj)
    p = p * y[i]
    if p < 1:
        beta[i] = beta[i] + y[i]

alpham = 1/T * (alpha.sum(axis = 0))

# calcular w2
for i in range(0, puntos):
    w2 = w2 + M[i, :] * alpham[i]

#predicciones
for i in range(puntos):
    p1[i] = np.sign(np.dot(M[i, :], w1))
    p2[i] = np.sign(np.dot(M[i, :], w2))
pre1 = sum(y ==p1) / puntos
pre2 = sum(y ==p2) / puntos


#Graficas
plt.figure(figsize=(7,7))
plt.subplot(2, 2, 1)
plt.title('Datos originales')
plt.scatter(X[y==1, 0], X[y==1, 1], color = 'blue')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color = 'red')

plt.subplot(2, 2, 3)
plt.title('Preceptron: ' + str(pre1))
plt.scatter(X[p1==1, 0], X[p1==1, 1], color = 'blue')
plt.scatter(X[p1==-1, 0], X[p1==-1, 1], color = 'red')

plt.subplot(2, 2, 4)
plt.title('SVM kernels: ' + str(pre2))
plt.scatter(X[p2==1, 0], X[p2==1, 1], color = 'blue')
plt.scatter(X[p2== -1, 0], X[p2== -1, 1], color = 'red')

plt.show()
