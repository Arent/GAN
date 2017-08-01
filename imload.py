import glob
from tqdm import tqdm
import Image
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np

def image_locations_and_labels_from_files(image_locations_filename, image_labels_filename):
	image_locations_file = open(image_locations_filename)
	image_labels_file = open(image_labels_filename)
	image_locations = image_locations_file.read().splitlines()
	image_labels = image_labels_file.read().splitlines()

	return image_locations, image_labels

image_locations, image_labels = image_locations_and_labels_from_files('data/image_locations','data/image_labels')
# convert string into tensors
tf_image_locations = ops.convert_to_tensor(image_locations, dtype=dtypes.string)
tf_image_labels = ops.convert_to_tensor(image_labels, dtype=dtypes.string)


image_locations_queue = tf.train.string_input_producer(image_locations)




#Define a subgraph that takes a filename, reads the file, decodes it, and                                                                                     
# enqueues it.                                                                                                                                                 
filename = image_locations_queue.dequeue()
label = image_locations_queue.dequeue()
image_bytes = tf.read_file(filename)
decoded_image = tf.image.decode_jpeg(image_bytes)
image_queue = tf.FIFOQueue(128, [tf.uint8], None)
enqueue_op = image_queue.enqueue(decoded_image)

# Create a queue runner that will enqueue decoded images into `image_queue`.                                                                                   
NUM_THREADS = 16
queue_runner = tf.train.QueueRunner(
	image_queue,
	[enqueue_op] * NUM_THREADS,  # Each element will be run from a separate thread.                                                                                       
	image_queue.close(),
	image_queue.close(cancel_pending_enqueues=True))

# Ensure that the queue runner threads are started when we call                                                                                               
# `tf.train.start_queue_runners()` below.                                                                                                                      
tf.train.add_queue_runner(queue_runner)

# Dequeue the next image from the queue, for returning to the client.                                                                                          
img = image_queue.dequeue()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	for i in tqdm(range(3)):
		image = img.eval()
		print(image.shape)
		print image
  		Image.fromarray(np.asarray(image)).show()

		