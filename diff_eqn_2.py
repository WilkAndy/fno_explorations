###################################################################
# Try to "solve the diffusion equation using a FNO"
# More exactly:
# - generate an initial condition and final condition that
#   corresponds to diffusion along a line.  This is easy, since
#   the diffusion equation is solved by the fundamental_solution
# - use a single-layer FNO to fit the initial and final conditions
# This is different than diff_eqn_1.py.  Here, since I know the
# Green's function for the diffusion equation, I use it in the
# FNO
# Green's fcn = exp(-x^2 / (4 cond t)) / sqrt(4 pi cond t)
# where cond is the conductivity,
# which has Fourier transform exp(-pi^2 4 cond t k^2)
# So, in the FNO i  multiply the lowest modes by exp(-width * k^2), where
# "width" is the only unknown in the problem.  I do not use an
# activation function, nor do i add a local linear transform
###################################################################

import os
import sys
import math
import tensorflow as tf
import matplotlib.pyplot as plt

###################################################################
# Generate the training set.
###################################################################
cond = 1.0
def fundamental_solution(x, t):
    return tf.math.exp(-tf.math.pow(x, 2) / 4.0 / cond / t) / tf.math.sqrt(4 * math.pi * cond * t)
num_samples = 100 # number of points to sample training set at (make sure this is even, for ft_size to work correctly)
xvals = tf.linspace(-10.0, 10.0, num_samples)
training_in = fundamental_solution(xvals, 1.0)
training_out = fundamental_solution(xvals, 2.0)

###################################################################
# Define the parameters in the single Fourier layer:
###################################################################
kmax = 12                                         # number of fourier modes to use in the fourier neural operator
width = tf.Variable(0, dtype = tf.complex64)      # the only unknown (2 real unknowns here, but we know the imaginary bit should be zero)
params = [width]
kvals = tf.constant([i for i in range(kmax)], dtype = tf.complex64)
def greens_fcn(k):
    return tf.math.exp(-tf.math.pow(k, 2) * width)

###################################################################
# Define the Fourier neural operator, the fourier layer, and the model
###################################################################
ft_size = num_samples // 2 + 1
all_zeroes = tf.constant([0] * (ft_size - kmax), dtype = tf.complex64)
def fno(u):
    # The Fourier Neural Operator:
    # (1) Takes a fourier transform, which gives a vector of size ft_size
    # (2) Element-wise multiplies the lowest-frequency modes by greens_fcn
    # (3) Appends these modes with zeroes to make a vector of ft_size
    # (4) Does the inverse fourier transformation
    return tf.signal.irfft(tf.concat([tf.math.multiply(greens_fcn(kvals), tf.signal.rfft(u)[:kmax]), all_zeroes], 0))
def fourier_layer(u):
    # The Fourier layer simply returns fno(u), without adding a local linear transform or using an activation function
    return fno(u)
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
epochs = 200
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
plt.legend()
plt.title("Can I use a Green's function in a FNO?")
plt.show()
plt.close()


sys.exit(0)
