import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from load_data import get_split, load_batch
from cngloss import group_loss
from inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
from param import *

import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
slim = tf.contrib.slim

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

with tf.Graph().as_default() as graph:
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = get_split('train', dataset_dir, file_pattern)
    images, _, labels = load_batch(dataset, batch_size=batch_size, height=image_size, width=image_size)

    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2(images, num_classes=dataset.num_classes)
        #end_points['Predictions'] = tf.nn.softmax(net, name='Predictions')

    variables_to_restore = slim.get_variables_to_restore(exclude = exclude_list)

    predictions = tf.argmax(end_points['Predictions'], 1)
    probabilities = end_points['Predictions']
    accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
    metrics_op = tf.group(accuracy_update, probabilities)

    #m_loss =  moment_loss(end_points['group_map'])
    m_loss = group_loss(end_points['group_map'])

    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)

    total_loss = tf.add(loss, tf.multiply(m_loss, 0.1))

    global_step = get_or_create_global_step()

    num_batches_per_epoch = int(train_img_num / batch_size)
    num_steps_per_epoch = num_batches_per_epoch

    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)


    lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('resnet_loss', loss)
    my_summary_op = tf.summary.merge_all()

    all_vars =  tf.trainable_variables()
    first_train_vars = [var for var in all_vars if not var.name.startswith(group_vars)]
    second_train_vars = [var for var in all_vars if var.name.startswith(group_vars)]
    
    optimizer_1 = tf.train.AdamOptimizer(learning_rate = lr)
    train_op_1 = slim.learning.create_train_op(loss, optimizer_1 , variables_to_train = first_train_vars)

    optimizer_2 = tf.train.AdamOptimizer(learning_rate = 1e-6)
    train_op_2 = slim.learning.create_train_op(m_loss, optimizer_2, variables_to_train = second_train_vars)

    def train_step(sess, train_op, global_step, flag):
        start_time = time.time()
        total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
        time_elapsed = time.time() - start_time

        if flag == 2:
            loss_flag = 'feature group'
        else:
            loss_flag = 'resnet'
        logging.info('global step %s: %s loss: %.4f (%.2f sec/step)', global_step_count, loss_flag, total_loss, time_elapsed)

        return total_loss, global_step_count

    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint_file)

    sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)

    with sv.managed_session() as sess:
        global_step_ = sess.run(sv.global_step) 

        for step in xrange(global_step_, num_epochs):
            if step % 100 == 0:
                #loss, _ = train_step(sess, train_op, sv.global_step)
	        summaries = sess.run(my_summary_op)
	        sv.summary_computed(sess, summaries)
            if step < 1000:
                loss, _ = train_step(sess, train_op_1, sv.global_step, flag=1)
            else:
                if (step % 2000) < 2000:

                    loss, _ = train_step(sess, train_op_1, sv.global_step, flag=1)
                else:
                    loss, _ = train_step(sess, train_op_2, sv.global_step, flag=2)

            if step % num_batches_per_epoch == 0:
	        logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
	        learning_rate_value, accuracy_value = sess.run([lr, accuracy])
	        logging.info('Current Learning Rate: %s', learning_rate_value)
	        logging.info('Current Streaming Accuracy: %s', accuracy_value)
	        logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits, probabilities, predictions, labels])
	        print'logits: \n', logits_value
	        print 'Probabilities: \n', probabilities_value
	        print 'predictions: \n', predictions_value
	        print 'Labels:\n:', labels_value


        logging.info('Final Loss: %s', loss)
        logging.info('Finished training! Saving model to disk now.')
        sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

