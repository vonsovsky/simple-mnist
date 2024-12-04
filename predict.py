from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

model = load_model('mnist_cnn_model.h5')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

im1 = x_test.reshape((x_test.shape[1234], 28, 28, 1)).astype('float32') / 255

pred = model.predict(im1)
print(pred)
