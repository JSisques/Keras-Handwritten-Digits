import os, ssl
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as kr
from keras.datasets import mnist #Importamos el dataset de imagenes
from keras.preprocessing.image import img_to_array, load_img


if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

#Numero de tipos a detectar
num_clases = 10
#Descripcion de la imagen en pixeles (28 x 28 x 1)
input_shape = (28, 28, 1)

#Cargamos el dataset con numeros escritos a mano
(x_train_original, y_train_original), (x_test_original, y_test_original) = mnist.load_data()

#Pasamos las clases a binario IMPORTANTE
y_train = kr.utils.to_categorical(y_train_original, num_clases) #00001, 00010, 00011 etc
y_test = kr.utils.to_categorical(y_test_original, num_clases) #00001, 00010, 00011 etc

#Escalamos las imagenes IMPORTANTE
x_train = x_train_original / 255
x_test = x_test_original / 255

print("x_train shape:", x_train.shape) #Comprobamos la forma que tiene
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


#Definimos el modelo
model = kr.Sequential()
model.add(kr.layers.Flatten(input_shape=(28, 28, 1)))
model.add(kr.layers.Dense(1000, activation='relu'))
model.add(kr.layers.Dense(500, activation='relu'))
model.add(kr.layers.Dense(10, activation='softmax'))

print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['acc', 'mse'],
)

model.fit(x_train, y_train, epochs=10)

model.save("model.h5")


score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

img_normal = x_test[4]
print(img_normal.shape)
img = np.expand_dims(img_normal,0)
print(img.shape)

predictions_single = model.predict(img)
print(np.argmax(predictions_single))

plt.imshow(img_normal)
plt.show()

predictions = model.predict(x_test)

print(np.argmax(predictions[0]))