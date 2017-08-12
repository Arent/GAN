import tensorflow as tf
import numpy as np
# import model.p

BATCH_SIZE    = 16
Z_DIMENSION = 100
# get training operations and placeholders

#Launch graph
with tf.Session() as sess:
	saver = tf.train.import_meta_graph('saved_models/2017-Aug-08-16-52-09/Epoch_1_Batch_0.ckpt.meta')
	saver.restore(sess,tf.train.latest_checkpoint('saved_models/2017-Aug-08-16-52-09/'))
	
	graph = tf.get_default_graph()
	z_batch = np.random.uniform(-1, 1, size=[BATCH_SIZE, Z_DIMENSION]).astype(np.float32)
	
	Z =tf.get_default_graph().get_tensor_by_name('Z:0')
	images = tf.get_default_graph().get_tensor_by_name('model/fake_images:0')


	print sess.run(images,feed_dict={Z:z_batch})

 