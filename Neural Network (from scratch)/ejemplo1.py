from Red_neuronal import *

# main
puntos = 100

X1 = np.random.random((puntos//2, 2))*5 + 5
X2 = np.random.random((puntos//2, 2))*5 + 10
X = np.concatenate((X1, X2), axis=0)
y = np.zeros(puntos)
np.random.shuffle(X)
for i in range(0, puntos):
    if 10 <= X[i, 0] and 10 <= X[i, 1]:
        y[i] = 1

nn = Neural_Net([2, 5, 1])
start = nn.predict(X)
start[start <= 0.5] = 0
start[start > 0.5] = 1
pre_start = sum(start == y) / puntos

hist = nn.train(X, y, 30000, reduct_step = 1)
hist_p = hist[0, :]
hist_l = hist[1, :]
predictions = nn.predict(X)
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1
pre = sum(predictions == y) / puntos


#Graficas
plt.figure(figsize=(7,7))
plt.subplot(2, 2, 1)
plt.title('Distribution of predictions')
plt.hist(nn.predict(X), bins = 10)
plt.subplot(2, 2, 2)
plt.title('Loss over time: ')
plt.plot(hist_l)
plt.subplot(2, 2, 3)
plt.title('Best prediction: ' + str(pre))
plt.scatter(X[predictions==1, 0], X[predictions==1, 1], color = 'blue')
plt.scatter(X[predictions== 0, 0], X[predictions== 0, 1], color = 'red')
plt.subplot(2, 2, 4)
plt.title('Presition over time')
plt.plot(hist_p)
plt.show()
