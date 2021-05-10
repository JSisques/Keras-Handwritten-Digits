import tensorflow.keras as kr
import numpy as np
import cv2

model = kr.models.load_model('./model.h5')

print(model.summary())

img = cv2.imread('./cero.png')

print(img)
print(img.shape)

img = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_CUBIC)

print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img.shape)

#cv2.imshow("image", img)
#cv2.waitKey(0)

img = np.expand_dims(img, 0)

print(img.shape)

predict = model.predict(img)
print(predict)
print(np.argmax(predict))