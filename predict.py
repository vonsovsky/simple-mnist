from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('mnist_cnn_model.h5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

im1 = x_test[1234].reshape((1, 28, 28, 1)).astype('float32') / 255

pred = model.predict(im1)
print(pred)
pred = np.argmax(pred, axis=1)
print(pred[0])