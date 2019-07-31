
# coding: utf-8

# In[1]:

import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras import metrics
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

get_ipython().magic('matplotlib inline')


# In[2]:

original_dim = 784
latent_dim = 2


# In[3]:

# encoder network
# using one Input Layer, one fully connected layer then fully connected to z_mean and z_log_sigma
inp = Input(shape = (original_dim, ))
X = Dense(256, activation = 'relu')(inp)
z_mean = Dense(latent_dim)(X)
z_log_sigma = Dense(latent_dim)(X)


# In[4]:

def sample_z(args):
    z_mean, z_log_sigma = args
    # a random value from a normal distribution with mean 0 and standard deviation 1
    epsilon = K.random_normal(shape = (K.shape(z_mean)[0], latent_dim),
                       mean = 0, stddev = 1)
    # main formula for sampling z from mean and covariance
    return z_mean + K.exp(z_log_sigma/2)*epsilon


# In[5]:

# Lambda(a lambda expression or a fn)(an array of args to the fn)
# this could have been done by simply using the fn but we use the Lambda layer so that z can easily be send as input to the next layer
z = Lambda(sample_z)([z_mean, z_log_sigma])


# In[6]:

# decoder network
# instantiate these layers separately so as to reuse them later
decoder_h = Dense(256, activation = 'relu')
X = decoder_h(z)
decoder_op = Dense(original_dim, activation = 'sigmoid')
X = decoder_op(X)


# In[7]:

vae = Model(inp, X)

encoder = Model(inp, z_mean)

# for defining the decoder (or generator network)
decoder_inp = Input(shape = (latent_dim,))
X = decoder_h(decoder_inp)
X = decoder_op(X)
generator = Model(decoder_inp, X)


# In[8]:

def vae_loss(X, X_decoded_mean):
    crossentropy_term = metrics.binary_crossentropy(X, X_decoded_mean)
    kl_divergence = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = -1)
    return crossentropy_term + kl_divergence


# In[9]:

# that is how we define a custom loss fn in keras
vae.compile(optimizer = 'rmsprop', loss = vae_loss)


# In[10]:

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


# In[11]:

vae.summary()


# In[ ]:

vae.fit(X_train, X_train, shuffle = True, epochs = 50, batch_size = 128, validation_data = (X_test, X_test))


# In[ ]:

X_test_encoded = encoder.predict(X_test, batch_size = 128)
plt.figure(figsize = (6, 6))
# scatter plot of both z_mean parameters with Y_test as colors of the points
plt.scatter(X_test_encoded[:, 0], X_test_encoded[:, 1], c = Y_test)
plt.colorbar()
plt.show()


# In[ ]:

n = 15
digit_size = 28
figure = np.zeros((digit_size*n, digit_size*n))

# linearly spaced 'n' numberss between -15 to 15
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

# enumerate generates a tuple of inedx and the value at that index for the whole list
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * epsilon_std
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

