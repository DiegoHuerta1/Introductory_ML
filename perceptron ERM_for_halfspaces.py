import random
import numpy as np
import matplotlib.pyplot as plt

puntos = 100
X = np.random.random((puntos, 2))*20  # 0  - 20
for i in range(puntos):
    if abs(X[i, 0] - X[i, 1]) < 1: 
        X[i, 0] = np.random.random()*10 #0 - 10
        X[i, 1] = X[i, 0] + (np.random.random()*9 + 1) # 11 - 20

w = np.zeros(2) # 2 pesos , 0-1
wp = np.array([-1, 1])
y = np.ones(puntos)*-1
mayor = np.zeros(2)

for i in range(0, puntos):
    if X[i, 0] - X[i, 1] < 0:
        y[i] = 1
        
    # encontrara max i ||x||^2
    if np.dot(X[i, :], X[i, :])> np.dot(mayor, mayor):
        mayor = X[i, :]


error = np.zeros(100)
M = 0 # errores totales
for j in range(0, 100):
    for i in range(0, puntos):
        if np.dot(w, X[i, :]) * y[i] <= 0:
            w = w + y[i] * X[i, :]
            error[j] = error[j]+1
            M = M+1
    if error[j] == 0:
            break

# Theorem (Agmon’54, Novikoff’62)
maxe = np.dot(wp, wp) * np.dot(mayor, mayor)
#prediccion
p = np.zeros(puntos)
for i in range(puntos):
    p[i] = np.sign(np.dot(w, X[i, :]))

#Graficas
xplot = np.array([0, 20])
plt.figure(figsize=(7,7))

plt.subplot(2, 2, 1)
plt.title('Datos originales')
plt.scatter(X[y==1, 0], X[y==1, 1], color = 'blue')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color = 'red')

plt.subplot(2, 2, 3)
plt.title('"w perfecta"')
plt.scatter(X[y==1, 0], X[y==1, 1], color = 'blue')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color = 'red')
yplot = -xplot*wp[0]/wp[1]
plt.plot(xplot, yplot, color = 'green')

plt.subplot(2, 2, 2)
plt.title('Prediccion final')
plt.scatter(X[p==1, 0], X[p==1, 1], color = 'blue')
plt.scatter(X[p==-1, 0], X[p==-1, 1], color = 'red')

plt.subplot(2, 2, 4)
plt.title('w aprendida')
plt.scatter(X[p==1, 0], X[p==1, 1], color = 'blue')
plt.scatter(X[p==-1, 0], X[p==-1, 1], color = 'red')
yplot = -xplot*w[0]/w[1]
plt.plot(xplot, yplot, color = 'green')

print('Terminado tras ' + str(j) + ' iteraciones')
print(str(M) + ' errores')
print('Maximos errores: ' + str(maxe))
print('Pesos: ' + str(w))
plt.show()
