import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from conv_layer import convolution2d
from pooling_layer import pooling
from upsampling_layer import upsampling_layer

#load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))    
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

#load trained weights (trained on MNIST using keras)
a = np.load("layer_weights.npy")

#set which image you would like to forward pass
inp_layer = x_train[4]
plt.imshow(inp_layer.reshape(28,28))
inp_layer = inp_layer.reshape(1,28,28)
print(inp_layer.shape)

#encoder architecture for autoencoder
print(inp_layer.shape)
conv_out1 = convolution2d(inp_layer, a[1][0], a[1][1].reshape(a[1][1].shape[0], 1), padding = 1, activation_type = "relu")
print(conv_out1.shape)
# plt.imshow(conv_out1[0, :, :].reshape(28,28))
pool_out1 = pooling(conv_out1, 2, stride = 2, type = "max")
print(pool_out1.shape)
# plt.imshow(pool_out1[15, :, :].reshape(14,14))
conv_out2 = convolution2d(pool_out1, a[3][0], a[3][1].reshape(a[3][1].shape[0], 1), padding = 1, activation_type = "relu")
print(conv_out2.shape)
# plt.imshow(conv_out2[0, :, :].reshape(14,14))
pool_out2 = pooling(conv_out2, 2, stride = 2, type = "max")
print(pool_out2.shape)
conv_out3 = convolution2d(pool_out2, a[5][0], a[5][1].reshape(a[5][1].shape[0], 1), padding = 1, activation_type = "relu")
plt.imshow(conv_out3[4, :, :].reshape(7,7))
print(conv_out3.shape)
pool_out3 = pooling(conv_out3, 2, stride = 2, padding = 1, type = "max")
print(pool_out3.shape)


#decoder
conv_out4 = convolution2d(pool_out3, a[7][0], a[7][1].reshape(a[7][1].shape[0], 1), padding = 1, activation_type = "relu")
print(conv_out4.shape)
upsample1  = upsampling_layer(conv_out4, size = (2,2), upsampling_type = "nearest_neighbour")
print(upsample1.shape)
conv_out5 = convolution2d(upsample1, a[9][0], a[9][1].reshape(a[9][1].shape[0], 1), padding = 1, activation_type = "relu")
print(conv_out5.shape)
upsample2  = upsampling_layer(conv_out5, size = (2,2), upsampling_type = "nearest_neighbour")
print(upsample2.shape)
conv_out6 = convolution2d(upsample2, a[11][0], a[11][1].reshape(a[11][1].shape[0], 1), padding = 0, activation_type = "relu")
print(conv_out6.shape)
upsample3  = upsampling_layer(conv_out6, size = (2,2), upsampling_type = "nearest_neighbour")
print(upsample3.shape)
conv_out7 = convolution2d(upsample3, a[13][0], a[13][1].reshape(a[13][1].shape[0], 1), padding = 1, activation_type = "sigmoid")
print(conv_out7.shape)


#compare output (original versus autoencoder reconstruction)
# display original
ax = plt.subplot(2, 1, 0 + 1)
plt.imshow(x_train[4].reshape(28, 28))
plt.gray()
ax.set_axis_off()

# display reconstruction
ax = plt.subplot(2, 1, 0 + 1 + 1)
plt.imshow(conv_out7.reshape(28, 28))
plt.gray()
ax.set_axis_off()
plt.show()