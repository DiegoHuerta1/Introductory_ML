import random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
puntos = 50
d = 3;
X1 = np.random.random((puntos//2, d-1))*5 + 5
X2 = np.random.random((puntos//2, d-1))*5 + 10
X = np.concatenate((X1, X2), axis=0)
y = np.ones(puntos) * -1

np.random.shuffle(X)
for i in range(0, puntos):
    if 10 <= X[i, 0] and 10 <= X[i, 1]:
        y[i] = 1

X = np.column_stack([X, np.ones(puntos).T])# extra feature por el bias, X.shape= puntos, d

p = np.zeros(puntos) #predict
p2 = np.zeros(puntos) #predict 2
w = np.zeros(d) # w[0] es el bias
M = 0 #Errores

A = np.eye(d) # (d, d)
eta = (d*d)/(d*d-1.0) # (1)
for t in range(0, puntos):
    x = X[t, :] # (d, 1)
    Ax = np.dot(A, x) # (d, 1)
    xtAx = np.dot(x, Ax) # (1)
    p[t] = np.sign(np.dot(w, x))
    if p[t] != y[t]:
        M = M+1
        w = w + (y[t]/(d+1)) * ((Ax)/(np.sqrt(xtAx)))
        A = eta*(A - (2.0/((d+1.0) * xtAx)) * (np.outer(Ax, Ax)))


pre = (sum(p==y))/puntos
print('Errores al entrenar: ' + str(M))
print('Presicion al entrenar: ' + str(pre))

for t in range(0, puntos):
    p2[t] = np.sign(np.dot(w, X[t, :]))
pre2 = (sum(p2==y))/puntos
print('Presicion con los valores w aprendidos: ' + str(pre2))
print('Valor w aprendido: ' + str(w))


plt.figure(figsize=(7,7))
plt.subplot(2, 2, 1)
plt.title('Datos originales')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color = 'blue')
plt.scatter(X[y == -1, 0], X[y == -1, 1], color = 'red')


plt.subplot(2, 2, 2)
plt.title('Prediccion mientras entrena')
plt.scatter(X[p == 1, 0], X[p == 1, 1], color = 'blue')
plt.scatter(X[p == -1, 0], X[p == -1, 1], color = 'red')

plt.subplot(2, 2, 4)
plt.title('Prediccion final')
xplot = np.array([5, 15])
yplot = -(w[2] + w[0] * xplot)/(w[1])
plt.scatter(X[p2 == 1, 0], X[p2 == 1, 1], color = 'blue')
plt.scatter(X[p2 == -1, 0], X[p2 == -1, 1], color = 'red')
plt.plot(xplot, yplot, color='green')
plt.xlim(5, 15)
plt.ylim(5, 15)
plt.show()
