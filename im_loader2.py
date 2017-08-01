# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import numpy as np 
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import Image


test_set_size = 5

IMAGE_HEIGHT  = 120
IMAGE_WIDTH   = 120
NUM_CHANNELS  = 3
BATCH_SIZE    = 5


def image_locations_and_labels_from_files(image_locations_filename, image_labels_filename):
	image_locations_file = open(image_locations_filename)
	image_labels_file = open(image_labels_filename)
	image_locations = image_locations_file.read().splitlines()
	image_labels = image_labels_file.read().splitlines()
	return image_locations, image_labels
  
all_filepaths, all_labels = image_locations_and_labels_from_files('data/image_locations','data/image_labels')
all_filepaths = all_filepaths
all_labels = all_labels


# convert string into tensors
all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.string)

# create a partition vector
partitions = [0] * len(all_filepaths)
partitions[:test_set_size] = [1] * test_set_size
random.shuffle(partitions)

# partition our data into a test and train set according to our partition vector
train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

# create input queues
train_input_queue = tf.train.slice_input_producer(
									[train_images, train_labels],
									shuffle=True)
test_input_queue = tf.train.slice_input_producer(
									[test_images, test_labels],
									shuffle=True)

# process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_image = tf.image.resize_images(train_image,[120,120])

train_label = train_input_queue[1]

file_content = tf.read_file(test_input_queue[0])
test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
test_image = tf.image.resize_images(test_image,[120,120])

test_label = test_input_queue[1]

# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


# collect batches of images before processing
train_image_batch, train_label_batch = tf.train.batch(
									[train_image, train_label],
									batch_size=BATCH_SIZE
									#,num_threads=1
									)
test_image_batch, test_label_batch = tf.train.batch(
									[test_image, test_label],
									batch_size=BATCH_SIZE
									#,num_threads=1
									)

print "input pipeline ready"

with tf.Session() as sess:
  
  # initialize the variables\
	sess.run(tf.local_variables_initializer())
  
  # initialize the queue threads to start to shovel data
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	print "from the train set:"
	for i in range(3):
		image_batch  = sess.run(train_image_batch)
		for img in image_batch:
			Image.fromarray(np.uint8(np.asarray(img))).show()


	print "from the test set:"
	for i in range(3):
		sess.run(test_image_batch)

  # stop our queue threads and properly close the session
	coord.request_stop()
	coord.join(threads)
	sess.close()