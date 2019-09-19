import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist


(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
#######################
def autoencoder_train(train_frame,validation_frame,input_size, layers,optimizer='adadelta',loss='binary_crossentropy',epoch_num=200,batch_size=256):
    input = Input(shape=(input_size,))
    encoded = Dense(layer_size[0], activation='relu')(input)
    for layer_iter in np.arange(1,len(layers)):
        encoded = Dense(layer_size[layer_iter], activation='relu')(encoded)
    ######
    decoded = Dense(layer_size[(len(layers)-1)], activation='relu')(encoded)
    for layer_iter in np.arange((len(layers)-2),1,-1):
        decoded = Dense(layer_size[layer_iter], activation='relu')(decoded)
    ######
    decoded = Dense(input_size, activation='sigmoid')(decoded)

    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    ######
    autoencoder.fit(train_frame, train_frame, epochs=epoch_num, batch_size=batch_size, \
                    shuffle=True, validation_data=(validation_frame, validation_frame))
    ######
    encoder = Model(input, encoded)

    return autoencoder,encoder
#######################
#######################
input_size = x_train.shape[1]
layer_size = [128,64,32]

autoencoder,encoder = autoencoder_train(train_frame=x_train,\
                                validation_frame=x_test, \
                                input_size=x_train.shape[1], \
                                layers=[128,64,32], \
                                optimizer='adadelta', \
                                loss='binary_crossentropy', \
                                epoch_num=30,batch_size=256)

encoded_imgs = encoder.predict(x_test)
encoded_imgs.shape

decoded_imgs = autoencoder.predict(x_test)
decoded_imgs.shape
####################
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
