import tensorflow as tf
import numpy as np
from param import batch_size, group_num

def group_loss(net):

    image_size = 8 

# distance loss
    net_flat = tf.reshape(net, [batch_size, image_size*image_size, group_num])
    max_index = tf.argmax(net_flat, axis=1)

    x_index = max_index % image_size
    y_index = max_index / image_size
    x_index = tf.expand_dims(tf.expand_dims(x_index, 1), 1)
    y_index = tf.expand_dims(tf.expand_dims(y_index, 1), 1)
    x_max = tf.image.resize_images(x_index, (image_size, image_size))
    y_max = tf.image.resize_images(y_index, (image_size, image_size))

    x_temp = np.zeros([batch_size, image_size, image_size, group_num])   
    y_temp = np.zeros([batch_size, image_size, image_size, group_num])   
    for index in range(image_size):
      x_temp[:,:,index,:] = index
      y_temp[:,index,:,:] = index
    x_p = tf.constant(x_temp, dtype=tf.float32)
    y_p = tf.constant(y_temp, dtype=tf.float32)

    x_y_diff = tf.cast(tf.pow(tf.subtract(x_max, x_p), 2) + tf.pow(tf.subtract(y_max, y_p), 2), tf.float32)
    #x_y_diff = tf.subtract(tf.maximum(x_y_diff,100.0), 100.0)
    dis_sum = tf.reduce_sum(tf.multiply(net, x_y_diff), [0,1,2,3])
    dis_sum = tf.divide(dis_sum, batch_size)
    dis_sum = tf.divide(dis_sum, 100)


# div loss
    margin = tf.constant(2e-4, shape=[1])
    list_0 = [tf.expand_dims(net[:,:,:,1], axis=3), tf.expand_dims(net[:,:,:,2], axis=3), tf.expand_dims(net[:,:,:,3], axis=3)]
    list_1 = [tf.expand_dims(net[:,:,:,0], axis=3), tf.expand_dims(net[:,:,:,2], axis=3), tf.expand_dims(net[:,:,:,3], axis=3)]
    list_2 = [tf.expand_dims(net[:,:,:,0], axis=3), tf.expand_dims(net[:,:,:,1], axis=3), tf.expand_dims(net[:,:,:,3], axis=3)]
    list_3 = [tf.expand_dims(net[:,:,:,0], axis=3), tf.expand_dims(net[:,:,:,1], axis=3), tf.expand_dims(net[:,:,:,2], axis=3)]
    diff0 = tf.subtract(tf.reduce_max(tf.concat(list_0, 3), axis=3), margin)
    diff1 = tf.subtract(tf.reduce_max(tf.concat(list_1, 3), axis=3), margin)
    diff2 = tf.subtract(tf.reduce_max(tf.concat(list_2, 3), axis=3), margin)
    diff3 = tf.subtract(tf.reduce_max(tf.concat(list_3, 3), axis=3), margin)

    div_temp = tf.add(tf.multiply(net[:,:,:,0], diff0), tf.add(tf.multiply(net[:,:,:,1], diff1), tf.add(tf.multiply(net[:,:,:,2], diff2), tf.multiply(net[:,:,:,3], diff3))))
    div_sum = tf.divide(tf.reduce_sum(div_temp, axis=[0,1,2]), batch_size)
    div_sum = tf.multiply(div_sum, 2)

    d_loss = tf.add(dis_sum, div_sum)
    #d_loss = dis_sum

    tf.summary.scalar('dis_sum', dis_sum)
    tf.summary.scalar('div_sum', div_sum)

    return d_loss
   


'''

    dis_list = list()
    for b in range(batch_size):
        for x_i in range(image_size):
            for y_i in range(image_size):
                for g in range(group_num):
                    x_max = x_index[b,g]
                    y_max = y_index[b,g]
                    distance = tf.cast(tf.add(tf.pow((x_i-x_max),2), tf.pow((y_i-y_max),2)), dtype=tf.float32)
                    dis_list.append(tf.multiply(net[b,x_max,y_max,g], distance))
    dis_sum = tf.reduce_sum(tf.stack(dis_list), 0)
    dis_sum = tf.divide(dis_sum, batch_size)

'''
    


