#################################################################
# I just want to see if I can code a Fourier Neural Operator
# into a single Fourier layer as per https://www.jmlr.org/papers/volume24/21-1524/21-1524.pdf
# and use tensorflow to find the layer's parameters to minimise a loss
#################################################################

import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

#################################################################
# Generate the training set.
# i just use one input function (training_in) and the Fourier layer
# can hopefully map that to the output function (training_out)
#################################################################
num_samples = 100 # number of points to sample training set at (make sure this is even, for ft_size to work correctly)
def training_fcn(x):
    return tf.where(x < 0, 0, tf.where(x < 1, x, tf.where(x < 2, 2 - x, 0)))
xvals = tf.linspace(-1.0, 4.0, num_samples)
training_in = training_fcn(xvals)
training_out = training_fcn(xvals - 1)

#################################################################
# Define the parameters in the single Fourier layer:
#################################################################
kmax = 12                                         # number of fourier modes to use in the fourier neural operator
r = tf.Variable([1] * kmax, dtype = tf.complex64) # linear transform on the kmax Fourier modes
w = tf.Variable(0, dtype = tf.float32)            # local linear transformation
params = [r, w]
# note: i choose to have no bias, because i think it would be
# b = tf.Variable([1] * num_samples, dtype = tf.float32) and that would over-parameterise this model
# that only has num_samples degrees of freedom!

#################################################################
# Define the Fourier neural operator, the fourier layer, and the model
#################################################################
ft_size = num_samples // 2 + 1
all_zeroes = tf.constant([0] * (ft_size - kmax), dtype = tf.complex64)
def fno(u):
    # The Fourier Neural Operator:
    # (1) Takes a fourier transform, which gives a vector of size ft_size
    # (2) Element-wise multiplies the lowest-frequency modes by r
    # (3) Appends these modes with zeroes to make a vector of ft_size
    # (4) Does the inverse fourier transformation
    return tf.signal.irfft(tf.concat([tf.math.multiply(r, tf.signal.rfft(u)[:kmax]), all_zeroes], 0))
def fourier_layer(u):
    # The Fourier layer adds the local linear transform to fno(u), and then applies the activation function
    return tf.math.maximum(0, tf.math.add(fno(u), w * u)) # relu activation
def model(u):
    # The model just has one fourier layer
    return fourier_layer(u)

#################################################################
# Define the loss, the optimizer, gradient descent, and train
#################################################################
def loss(u):
    return tf.reduce_mean(tf.square(training_out - model(u)))
optimizer = tf.keras.optimizers.Adam()
def gradient_descent():
    with tf.GradientTape(persistent = True) as tp:
        epoch_loss = loss(training_in)
    gradient = tp.gradient(epoch_loss, params)
    del tp
    optimizer.apply_gradients(zip(gradient, params))
    return epoch_loss
epochs = 10000
for epoch in range(epochs):
    epoch_loss = gradient_descent()
    print("epoch =", epoch, "loss =", epoch_loss.numpy())


#################################################################
# Plot some results
#################################################################
plt.figure()
plt.plot(xvals, training_in, 'k:', label = 'training: initial')
plt.plot(xvals, training_out, 'k--', label = 'training: final')
plt.plot(xvals, fourier_layer(training_in), 'k', label = 'FNO on training set')
#plt.plot(xvals, tf.math.maximum(0, tf.math.sin(xvals)), 'r:', label = 'test: initial')
#plt.plot(xvals, tf.math.maximum(0, tf.math.sin(xvals - 1)), 'r--', label = 'test: final')
#plt.plot(xvals, fourier_layer(tf.math.maximum(0, tf.math.sin(xvals))), 'r', label = 'FNO on test set')
plt.legend()
plt.title("Can I find a single Fourier layer that takes training initial to final?")
plt.show()
plt.close()


sys.exit(0)
