from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_test.shape)
#plt.figure()
#plt.imshow(x_test[1234])
#plt.savefig('test/8_1.png')
#plt.show()

#plt.figure()
#plt.imshow(x_test[2558])
#plt.savefig('test/5_1.png')
#plt.show()

#plt.figure()
#plt.imshow(x_test[6378])
#plt.savefig('test/4_1.png')
#plt.show()
