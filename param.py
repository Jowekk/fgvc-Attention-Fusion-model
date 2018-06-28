#---------------dataset information-------------#
dataset_dir = './dataset'
labels_file = './dataset/labels.txt'
file_pattern = 'fgvc_%s.tfrecord'
image_size = 299
num_classes = 100
test_img_num = 3333
train_img_num = 6667
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
    'label': 'A label that is as such -- 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
}


#----------------log and checkpoint------------#
log_dir = './log'
checkpoint_file = './checkpoints/model.ckpt-0'#'./checkpoints/inception_resnet_v2_2016_08_30.ckpt'
exclude_list =  ['InceptionResnetV2/Logits', 'InceptionResnetV2/Location', 'InceptionResnetV2/group_map']
group_vars = 'InceptionResnetV2/group_map'


#----------------net information---------------#
dropout_keep_prob = 0.8

#----------------train information-------------#
initial_learning_rate = 2e-4
learning_rate_decay_factor = 0.95
num_epochs_before_decay = 2

batch_size = 16
num_epochs = 60000

num_batches_per_epoch = int(train_img_num / batch_size)
num_steps_per_epoch = num_batches_per_epoch 
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

#--------------feature maps group infromation----#
group_num = 4


