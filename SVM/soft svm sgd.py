import numpy as np
import matplotlib.pyplot as plt
from random import randrange

# Orden de los puntos
puntos = 100
s = 1   # separacion de los datos
inc= 0.8   # inclinacion de la frontera de desicion


d=2
X = np.random.random((puntos, 2))*20  # 0  - 20
for i in range(puntos):
    if abs(inc*X[i, 1] - X[i, 0]) < s: # no cumple la separacion
        X[i, 0] = np.random.random()*(10-s) #0 - 10-s
        X[i, 1] = (np.random.random()*(10-s)) + (10+s) # 10+s - 20

w = np.zeros(2) # 2 pesos , 0-1
wp = np.array([-1, 1*inc])
y = np.ones(puntos)*-1

for i in range(0, puntos):
    if X[i, 0] - inc*X[i, 1] < 0:
        y[i] = 1

# algoritmo
T = 10000
theta = np.zeros(d)
wt = np.zeros(d)
for t in range(1, T):
    wt = theta/t
    w = w + wt
    m = np.random.randint(puntos)
    if y[m] * np.dot(wt, X[m, :]) < 1:
        theta = theta + y[m] * X[m, :]
w = 1/(T) * w


#prediccion
p = np.zeros(puntos)
for i in range(puntos):
    p[i] = np.sign(np.dot(w, X[i, :]))
pre = (sum(p==y))/puntos

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
plt.xlim(0, 20)
plt.ylim(0, 20)

plt.subplot(2, 2, 2)
plt.title('Prediccion final:' + str(pre))
plt.scatter(X[p==1, 0], X[p==1, 1], color = 'blue')
plt.scatter(X[p==-1, 0], X[p==-1, 1], color = 'red')


plt.subplot(2, 2, 4)
plt.title('w aprendida')
plt.scatter(X[y==1, 0], X[y==1, 1], color = 'blue')
plt.scatter(X[y==-1, 0], X[y==-1, 1], color = 'red')
yplot = -xplot*w[0]/w[1]
plt.plot(xplot, yplot, color = 'green')
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()
