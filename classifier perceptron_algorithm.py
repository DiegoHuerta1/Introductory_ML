import random
import numpy as np
import matplotlib.pyplot as plt

puntos = 50
A = np.random.random((puntos, 2))*5 + 5  # 5  - 10
B = np.random.random((puntos, 2))*5 + 10 # 10 - 15

w = np.random.random(3) # 3 pesos , 0-1

M = np.concatenate((A, B), axis=0)
M = np.column_stack([np.ones(puntos*2).T, M])# extra feature por el bias
t = np.concatenate((np.ones(puntos).T, -1*np.ones(puntos).T), axis = 0) # labels

error = np.zeros(100)

for j in range(0, 100):
    for i in range(0, puntos):
        if np.dot(w, M[i, :]) * t[i] <= 0:
            w = w + t[i] * M[i, :]
            error[j] = error[j]+1
        if np.dot(w, M[i+puntos, :]) * t[i+puntos] <= 0:
            w = w + t[i+puntos] * M[i+puntos, :]
            error[j] = error[j]+1
    if error[j] == 0:
            break


plt.scatter(A[:, 0], A[:, 1], color = 'blue')
plt.scatter(B[:, 0], B[:, 1], color = 'red')
x = np.arange(5,16,5)
y = -(w[0] + w[1] * x)/(w[2])
plt.plot(x, y, color='green')
plt.xlim(5, 15)
plt.ylim(5, 15)
print('Terminado tras ' + str(j) + ' iteraciones')
print('Pesos: ' + str(w))
plt.show()
