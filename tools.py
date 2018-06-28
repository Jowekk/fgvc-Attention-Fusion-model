import tensorflow as tf
import numpy as np

def gauss(x):

  # Gaussian Filter
  K = np.array([[0.003765,0.015019,0.023792,0.015019,0.003765],
  [0.015019,0.059912,0.094907,0.059912,0.015019],
  [0.023792,0.094907,0.150342,0.094907,0.023792],
  [0.015019,0.059912,0.094907,0.059912,0.015019],
  [0.003765,0.015019,0.023792,0.015019,0.003765]], dtype='float32')

  w = tf.constant(K.reshape(K.shape[0],K.shape[1], 1, 1))

  lowpass = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

  return lowpass

def norm(x):
  
  xmin, xmax = tf.reduce_max(x), tf.reduce_min(x)
  x = tf.divide(x-xmin, xmax-xmin)
  return x
