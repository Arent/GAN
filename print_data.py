
import tensorflow as tf
import numpy as np
import Image
from hyper_parameters import *
from image_loader import *


train_image_batch, train_label_batch, test_image_batch, test_label_batch = get_batches()



with tf.Session() as sess:
  
  # initialize the variables\
	sess.run(tf.local_variables_initializer())
  
  # initialize the queue threads to start to shovel data
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	print "from the train set:"
	for i in range(5):
		image_batch  = sess.run(train_image_batch)
		for img in image_batch:
			Image.fromarray(np.uint8(np.asarray(img))).show()


	print "from the test set:"
	for i in range(5):
		image_batch  = sess.run(test_image_batch)
		for img in image_batch:
			Image.fromarray(np.uint8(np.asarray(img))).show()
  # stop our queue threads and properly close the session
	coord.request_stop()
	coord.join(threads)
	sess.close()