from Red_neuronal import *
from math import pi
from math import sin

X = np.arange(0.5, pi-0.5, 0.01)
Y = np.sin(X)

nn = Neural_Net([1, 5, 5, 5, 1])
hist = nn.train(X, Y, 200000, reduct_step = 100)
hist_l = hist[1, :]

predictions = nn.predict(X)

#Graficas
plt.figure(figsize=(7,7))
plt.subplot(2, 2, 1)
plt.title('Data')
plt.plot(X, Y)
plt.subplot(2, 2, 2)
plt.title('Predictions')
plt.plot(X, predictions)
plt.subplot(2, 2, 3)
plt.title('Loss over time: ')
plt.plot(hist_l)

plt.show()
