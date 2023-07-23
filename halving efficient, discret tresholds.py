import random
import numpy as np
import matplotlib.pyplot as plt

puntos = 200
n = 100 # valores van: 0, 1/n, 2/n, 3/n, ... n/n

X = np.ones(puntos)
y = np.ones(puntos) * -1

mas = 0
treshold = random.random()

for i in range(0, puntos):
    X[i] = random.randint(0, n) / n
    if  (treshold) < X[i] and X[i] <= 1 :
        y[i] = 1
        mas = mas+1

plt.scatter(X[y == 1], np.ones(mas), color = 'red', label = 'y=1')
plt.scatter(X[y == -1], np.ones(puntos - mas), color = 'blue', label = 'y=-1')
plt.legend()
#plt.show()

l = -0.5 /n
r = 1 + (0.5/n)

p = np.ones(puntos)

for i in range(0, puntos):
    x = X[i]
    p[i] = np.sign((x-l)-(r-x))
    if l < x and x < r :
        if y[i] == 1:
            r = x - 0.5/n
        if y[i] == -1:
            l = x + 0.5/n
            
er = puntos - sum(p == y)
print('l = ' + str(l))
print('r = ' + str(r))
print('Treshold aprendido = ' + str((l+r)/2))
print('Treshold real = ' + str(treshold))
print('Errores mientras aprendia: ' + str(er))

    
