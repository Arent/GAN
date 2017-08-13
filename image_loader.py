# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import numpy as np 
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import Image
from hyper_parameters import *

def image_locations_and_labels_from_files(image_locations_filename, image_labels_filename):
	image_locations_file = open(image_locations_filename)
	image_labels_file = open(image_labels_filename)
	image_locations = image_locations_file.read().splitlines()
	image_labels = image_labels_file.read().splitlines()
	return image_locations, image_labels
 


def get_batches():
	all_filepaths, all_labels = image_locations_and_labels_from_files(PATH_FOLDER,LABEL_FOLDER)
	all_filepaths = all_filepaths #DEBUG
	all_labels = all_labels #DEBUG


	# convert string into tensors
	all_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
	all_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.string)

	# create a partition vector
	partitions = [0] * len(all_filepaths)
	partitions[:TEST_SET_SIZE] = [1] * TEST_SET_SIZE

	random.shuffle(partitions)

	# partition our data into a test and train set according to our partition vector
	train_images, test_images = tf.dynamic_partition(all_images, partitions, 2)
	train_labels, test_labels = tf.dynamic_partition(all_labels, partitions, 2)

	number_of_train_files = len(partitions) - TEST_SET_SIZE
	number_of_test_files = TEST_SET_SIZE


	# create input queues
	train_input_queue = tf.train.slice_input_producer(
										[train_images, train_labels],
										shuffle=True,
										num_epochs = EPOCHS)
	test_input_queue = tf.train.slice_input_producer(
										[test_images, test_labels],
										shuffle=True,
										num_epochs= EPOCHS)

	# process path and string tensor into an image and a label
	file_content = tf.read_file(train_input_queue[0])
	train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)

	# pre processing, image dimensions and range
	train_image = tf.image.resize_images(train_image,[IMAGE_HEIGHT,IMAGE_WIDTH])
	train_image = ((train_image /255) -1) * 2 # rescale to -1 , 1
	train_label = train_input_queue[1]

	file_content = tf.read_file(test_input_queue[0])
	test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
	test_image = tf.image.resize_images(test_image,[IMAGE_HEIGHT,IMAGE_WIDTH])
	test_image = ((test_image /255) -1) * 2 # rescale to -1 , 1

	test_label = test_input_queue[1]

	# define tensor shape
	train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
	test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


	# collect batches of images before processing
	train_image_batch, train_label_batch = tf.train.batch(
										[train_image, train_label],
										batch_size=BATCH_SIZE
										,num_threads=1
										)
	test_image_batch, test_label_batch = tf.train.batch(
										[test_image, test_label],
										batch_size=BATCH_SIZE
										,num_threads=1
										)
	return number_of_train_files, number_of_test_files, train_image_batch\
	, train_label_batch, test_image_batch, test_label_batch


