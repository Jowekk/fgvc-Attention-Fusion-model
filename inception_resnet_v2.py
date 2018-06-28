# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of the Inception Resnet V2 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from expressway import expressway_net
import tensorflow as tf
from param import group_num, batch_size
from tools import gauss, norm

slim = tf.contrib.slim

# my_code

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name, is_training):
  with tf.variable_scope(layer_name) :

    squeeze = slim.avg_pool2d(input_x, input_x.get_shape()[1:3], padding='VALID', scope=layer_name+'AvgPool')

    excitation = slim.fully_connected(squeeze, int(out_dim/ratio), activation_fn=None, scope=layer_name+'_fully_connected1')
    excitation = slim.dropout(excitation, 0.5, is_training=is_training, scope=layer_name + 'Dropout_1')
    excitation = tf.nn.relu(excitation, name = layer_name + '_relu')
    excitation = slim.fully_connected(excitation, int(out_dim),  activation_fn=None, scope=layer_name+'_fully_connected2')
    excitation = slim.dropout(excitation, 0.5, is_training=is_training, scope=layer_name + 'Dropout_2')
    excitation = tf.nn.sigmoid(excitation, name = layer_name + '_sigmoid')

    excitation = tf.reshape(excitation, [-1,1,1,out_dim])
    scale = input_x * excitation
    group_net = tf.reduce_mean(scale, axis=3, keep_dims=True)
    group_net = gauss(group_net)
   
    group_net = slim.flatten(group_net) 
    temp_list = list()
    for i in range(batch_size):
      temp_img = group_net[i,:]
      temp_img = tf.reshape(norm(temp_img), (8,8))
      temp_img = tf.expand_dims(temp_img, axis=0)
      temp_list.append(temp_img)
    group_net = tf.concat(temp_list, axis=0)
    group_net = tf.expand_dims(group_net, axis=3)
    return scale, group_net

# my_code

def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def inception_resnet_v2(inputs, num_classes=1001, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=tf.AUTO_REUSE,
                        scope='InceptionResnetV2'):
  """Creates the Inception Resnet V2 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """
  end_points = {}

  with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse):
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                          stride=1, padding='SAME'):

        # 149 x 149 x 32
        net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                          scope='Conv2d_1a_3x3')
        end_points['Conv2d_1a_3x3'] = net
        # 147 x 147 x 32
        net = slim.conv2d(net, 32, 3, padding='VALID',
                          scope='Conv2d_2a_3x3')
        end_points['Conv2d_2a_3x3'] = net
        # 147 x 147 x 64
        net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
        end_points['Conv2d_2b_3x3'] = net
        # 73 x 73 x 64
        net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                              scope='MaxPool_3a_3x3')
        end_points['MaxPool_3a_3x3'] = net
        # 73 x 73 x 80
        net = slim.conv2d(net, 80, 1, padding='VALID',
                          scope='Conv2d_3b_1x1')
        end_points['Conv2d_3b_1x1'] = net
        # 71 x 71 x 192
        net = slim.conv2d(net, 192, 3, padding='VALID',
                          scope='Conv2d_4a_3x3')
        end_points['Conv2d_4a_3x3'] = net
        # 35 x 35 x 192
        net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                              scope='MaxPool_5a_3x3')
        end_points['MaxPool_5a_3x3'] = net
        
        # 35 x 35 x 320
        with tf.variable_scope('Mixed_5b'):
          with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
          with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                        scope='Conv2d_0b_5x5')
          with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                        scope='Conv2d_0c_3x3')
          with tf.variable_scope('Branch_3'):
            tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                         scope='AvgPool_0a_3x3')
            tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                       scope='Conv2d_0b_1x1')
          net = tf.concat(axis=3, values=[tower_conv, tower_conv1_1,
                              tower_conv2_2, tower_pool_1])

        end_points['Mixed_5b'] = net
        net = slim.repeat(net, 10, block35, scale=0.17)

        # 17 x 17 x 1024
        with tf.variable_scope('Mixed_6a'):
          with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                     scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                        stride=2, padding='VALID',
                                        scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_2'):
            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                         scope='MaxPool_1a_3x3')
          net = tf.concat(axis=3, values=[tower_conv, tower_conv1_2, tower_pool])

        end_points['Mixed_6a'] = net
        net = slim.repeat(net, 20, block17, scale=0.10)

        with tf.variable_scope('Mixed_7a'):
          with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                       padding='VALID', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_1'):
            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                        padding='VALID', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_2'):
            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                        scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                        padding='VALID', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_3'):
            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                         scope='MaxPool_1a_3x3')
          net = tf.concat(axis=3, values=[tower_conv_1, tower_conv1_1,
                              tower_conv2_2, tower_pool])

        end_points['Mixed_7a'] = net 
        
        net = slim.repeat(net, 9, block8, scale=0.20) #TODO
        net = block8(net, activation_fn = None)

        net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
        end_points['Conv2d_7b_1x1'] = net

# my_code
        with tf.variable_scope('group_map'):
          net_list = list()
          group_list = list()
          for i in range(group_num):
            scale,temp_net = Squeeze_excitation_layer(net[:,:,:,i*384:(i+1)*384], out_dim=384, ratio=2, layer_name='group_'+str(i), is_training=is_training)
            net_list.append(scale)
            group_list.append(temp_net)
        net_group = tf.concat(group_list, axis=3)
        net = tf.concat(net_list, axis=3)
        end_points['group_map'] = net_group
        tf.summary.image('net_group', net_group, max_outputs=4)

        with tf.variable_scope('Location'):
          loca_list = list()
          for index in range(group_num):
            group_A = net[:,:,:,index*384:(index+1)*384]
            feature_A = tf.expand_dims(net_group[:,:,:,index], axis=3)
            for except_i in range(group_num):
              if except_i != index:
                group_B = net[:,:,:,except_i*384:(except_i+1)*384]
                feature_B = tf.expand_dims(net_group[:,:,:,except_i], axis=3)

                f_A_and_B = tf.concat([feature_A, feature_B], axis=3)
                f_A_and_B = tf.reduce_sum(f_A_and_B, axis=3, keep_dims=True)

                group_A = slim.conv2d(group_A, 384, 1, scope='Loca_A_' + str(index) + '_' + str(except_i))
                group_B = slim.conv2d(group_B, 384, 1, scope='Loca_B_' + str(index) + '_' + str(except_i))             
                A_multiply_B = tf.multiply(group_A, group_B)
              
                A_relative_B = tf.multiply(f_A_and_B, A_multiply_B)
                A_relative_B = slim.conv2d(A_relative_B, 384, 1, scope='A_B_' + str(index) + '_' + str(except_i))
                loca_list.append(A_relative_B)
        
        net = tf.concat(loca_list, axis=3)
# my_code
        with tf.variable_scope('Logits'):
          # bitch_size * 8 * 8 * 1536
          end_points['PrePool'] = net
          net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                scope='AvgPool_1a_8x8')
          net = slim.flatten(net)

          net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='Dropout')

          end_points['PreLogitsFlatten'] = net

          logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                        scope='Logits')
          end_points['Logits'] = logits
          end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points

inception_resnet_v2.default_image_size = 299



def inception_resnet_v2_arg_scope(weight_decay=0.00004,
                                  batch_norm_decay=0.9997,
                                  batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_resnet_v2.

  Args:
    weight_decay: the weight decay for weights variables.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Returns:
    a arg_scope with the parameters needed for inception_resnet_v2.
  """
  # Set weight_decay for weights in conv2d and fully_connected layers.
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
    }
    # Set activation_fn and parameters for batch_norm.
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params) as scope:
      return scope
