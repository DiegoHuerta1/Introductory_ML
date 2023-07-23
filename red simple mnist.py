from mnist_cargar import *  # para cargar el mnist data set

(X_train,y_train),(X_test,y_test)= load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)

# linea base de MLP para MNIST dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
# semilla de reproducibilidad
seed = 7
numpy.random.seed(seed)

num_pixels = 28 * 28
num_classes = 10

# define la linea base del modelo
def baseline_model():
  # crea el modelo
  model = Sequential()
  model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal',
activation='relu'))
  model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
  # Compila el modelo
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# Construir del modelo
model = baseline_model()
# Ajusta el modelo
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200,
verbose=2)
# Final evaluacion del modelo
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
