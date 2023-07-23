import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Red neuronal con tf, enfoque de bajo nivel (algebra lineal)
# Con estructura [2, 3, 1]

#Datos
puntos = 300
X_1 = np.random.random((puntos, 2))*4-2  # -2 a 2
y_1 = np.zeros(puntos)
X_2 = np.random.random((puntos, 2))*4-2  # -2 a 2
y_2 = np.zeros(puntos)

for i in range(puntos):
    if 1.5 <= X_1[i, 0]*X_1[i, 0] + X_1[i, 1]*X_1[i, 1] <= 2: # margen
        X_1[i, 0] = np.random.random()
        X_1[i, 1] = np.random.random()
        y_1[i] = 1
    elif X_1[i, 0]*X_1[i, 0] + X_1[i, 1]*X_1[i, 1] < 1.5: # dentro
        y_1[i] = 1

    if 1.5 <= X_2[i, 0]*X_2[i, 0] + X_2[i, 1]*X_2[i, 1] <= 2: # margen
        X_2[i, 0] = np.random.random()
        X_2[i, 1] = np.random.random()
        y_2[i] = 1
    elif X_2[i, 0]*X_2[i, 0] + X_2[i, 1]*X_2[i, 1] < 1.5: # dentro
        y_2[i] = 1

X_train = tf.constant(X_1, tf.float32, shape = (puntos, 2))
y_train = tf.constant(y_1, tf.float32, shape = (puntos, 1))

X_test = tf.constant(X_2, tf.float32, shape = (puntos, 2))
y_test = tf.constant(y_2, tf.float32, shape = (puntos, 1))


#Neural Network
# Definir parametros como variables
w1 = tf.Variable(tf.random.normal([2, 3]))
b1 = tf.Variable(tf.ones(3), dtype = tf.float32)

w2 = tf.Variable(tf.random.normal([3, 1]))
b2 = tf.Variable([0], dtype = tf.float32)

# Definir el modelo, (predicciones y funcion de perdida)
def model(w1, b1, w2, b2, features = X_train):
    #devuelve la prediccion, tiene dropout
    layer1 = tf.keras.activations.relu(tf.matmul(features, w1)+b1)
    dropout = tf.keras.layers.Dropout(0.1)(layer1)
    return tf.keras.activations.sigmoid(tf.matmul(dropout, w2)+b2)

def loss_function(w1, b1, w2, b2, features=X_train, targets=y_train):
    predictions = model(w1, b1, w2, b2)
    return tf.keras.losses.binary_crossentropy(targets, predictions)

#Entrenar
opt = tf.keras.optimizers.SGD()
for j in range(1000):
    opt.minimize(lambda: loss_function(w1, b1, w2, b2), var_list = [w1, b1, w2, b2])
    print(f'Epoch {j}: loss', end=' ')
    print(sum(loss_function(w1, b1, w2, b2).numpy()))

#predicciones
p_1 = model(w1, b1, w2, b2, X_train).numpy().T[0]
p_b1 = np.round(p_1)
acuracy1 = sum(p_b1 == y_1)/puntos

p_2 = model(w1, b1, w2, b2, X_test).numpy().T[0]
p_b2 = np.round(p_2)
acuracy2 = sum(p_b2 == y_2)/puntos

print(confusion_matrix(y_test, p_b2))

#Graficas
fig, ax = plt.subplots(2, 2)
ax[0, 0].scatter(X_1[y_1==1, 0], X_1[y_1==1, 1], color = 'blue')
ax[0, 0].scatter(X_1[y_1==0, 0], X_1[y_1==0, 1], color = 'red')
ax[0, 0].set_title('Datos de entrenamiento')

ax[0, 1].scatter(X_1[p_1>=0.5, 0], X_1[p_1>=0.5, 1], color = 'blue')
ax[0, 1].scatter(X_1[p_1<0.5, 0], X_1[p_1<0.5, 1], color = 'red')
ax[0, 1].set_title('Predicciones '+str(np.round(acuracy1, decimals = 2)))

ax[1, 0].scatter(X_2[y_2==1, 0], X_2[y_2==1, 1], color = 'blue')
ax[1, 0].scatter(X_2[y_2==0, 0], X_2[y_2==0, 1], color = 'red')
ax[1, 0].set_title('Datos de prueba')

ax[1, 1].scatter(X_2[p_2>=0.5, 0], X_2[p_2>=0.5, 1], color = 'blue')
ax[1, 1].scatter(X_2[p_2<0.5, 0], X_2[p_2<0.5, 1], color = 'red')
ax[1, 1].set_title('Predicciones de prueba '+str(np.round(acuracy2, decimals = 2)))

plt.show()
