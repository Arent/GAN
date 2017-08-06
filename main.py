import tensorflow as tf
import numpy as np
import Image
from hyper_parameters import *
from image_loader import *
from model import *



train_image_batch, train_label_batch, test_image_batch, test_label_batch = get_batches()
Z, real_images, probability_real, logit_real, probability_fake, logit_fake\
,loss_discriminator_real, loss_discriminator_fake, loss_discriminator ,loss_generator = build_model()





with tf.Session() as sess:

  	discriminator_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/discriminator')
	generator_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/generator')

	train_op_discrim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_ADAM)\
							.minimize(loss_discriminator, var_list=discriminator_variables, name='D_step')
	train_op_gen = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_ADAM)\
							.minimize(loss_generator, var_list=generator_variables, name='G_step')

  # initialize the variables
	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer())
							
  # initialize the queue threads to start to shovel data
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	iBatch = 0
	try:
		while not coord.should_stop():
			print(iBatch)
			z_batch = np.random.uniform(-1, 1, size=[BATCH_SIZE, Z_DIMENSION]).astype(np.float32)
			image_batch  = sess.run(train_image_batch)

			_, loss_generator_local = sess.run([train_op_gen, loss_generator], feed_dict={Z:z_batch, real_images:image_batch})
			_, loss_discriminator_local = sess.run([train_op_discrim, loss_discriminator], feed_dict={Z:z_batch, real_images:image_batch})

			iBatch = iBatch + 1
	
	except Exception, e:
	# Report exceptions to the coordinator.
		coord.request_stop(e)
	
	finally:
  		# stop our queue threads and properly close the session
		coord.request_stop()
		coord.join(threads)
		sess.close()