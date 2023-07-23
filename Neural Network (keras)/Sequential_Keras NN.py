import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping

# Red neuronal con tf, enfoque de nivel medio (Sequential model)
# Estructura [2, 3, 3, 1]

# Modelo entrenado en: Sequential_Keras NN_class.h5

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
model = keras.Sequential()

model.add(keras.layers.Dense(3, activation='sigmoid', input_shape=(2,)))
model.add(keras.layers.Dense(3, activation='sigmoid'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.5), loss= 'binary_crossentropy', metrics = ['accuracy'])


early_stop = EarlyStopping(patience=20)
model_training = model.fit(X_train, y_train, epochs = 700, validation_data=(X_test, y_test), callbacks = [early_stop])

model.save('Sequential_Keras NN.h5')

#predicciones
p_1 = model.predict(X_train)[:, 0]
acuracy1 = model.evaluate(X_train, y_train)[1] # la primera es perdida, la segunda es presicion

p_2 = model.predict(X_test)[:, 0]
acuracy2 = model.evaluate(X_test, y_test)[1]

y_pred = np.round(p_2)
print(confusion_matrix(y_test, y_pred))

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
