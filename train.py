import tensorflow as tf
import numpy as np
import Image
from hyper_parameters import *
from image_loader import *
from model import *
import datetime
import os
import math

run_identifier = '{:%Y-%b-%d-%H-%M-%S}'.format(datetime.datetime.now())

def create_save_location():
	assert os.path.isdir("saved_models/"+run_identifier) == False
	
	os.mkdir("saved_models/" + run_identifier) 
	os.mkdir("saved_models/"+run_identifier+"/samples")
	return "saved_models/" + run_identifier+ "/"

save_location = create_save_location()


n_train_files, n_test_files\
,train_image_batch, train_label_batch, test_image_batch, test_label_batch = get_batches()

batches_per_epoch = int(math.ceil(float(n_train_files) / BATCH_SIZE))

Z, real_images, probability_real, logit_real, probability_fake, logit_fake\
,loss_discriminator_real, loss_discriminator_fake, loss_discriminator ,loss_generator, merged_summary_op = build_model()



with tf.Session() as sess:
	saver = tf.train.Saver(max_to_keep=10)
	summary_writer = tf.summary.FileWriter('tensorboard_logs', tf.get_default_graph())

	discriminator_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/discriminator')
	generator_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/generator')

	#optimiser = tf.train.AdamOptimizer(LEARNING_RATE, bet1=BETA_ADAM)
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
			epoch = int(iBatch / batches_per_epoch) + 1
			batch_number = iBatch % batches_per_epoch

			print('Epoch: '+str(epoch) +'/'+str(EPOCHS) + ' Batch: ' + str(batch_number) +'/' + str(batches_per_epoch -1))
			
			z_batch = np.random.uniform(-1, 1, size=[BATCH_SIZE, Z_DIMENSION]).astype(np.float32)
			image_batch  = sess.run(train_image_batch)

			_, loss_generator_local = sess.run([train_op_gen, loss_generator], feed_dict={Z:z_batch, real_images:image_batch})
			_, loss_discriminatorator_local = sess.run([train_op_discrim, loss_discriminator], feed_dict={Z:z_batch, real_images:image_batch})
			
			

			summary_str = sess.run(merged_summary_op, feed_dict={Z:z_batch, real_images:image_batch})
			summary_writer.add_summary(summary_str, iBatch)

			if batch_number % (batches_per_epoch / 2 ) == 0:
				saved_path = saver.save(sess, save_location +"Epoch_"+str(epoch) + "_Batch_" +str(batch_number)+".ckpt")
				print("Model saved in file: %s" % saved_path)
			iBatch = iBatch + 1
	
	except Exception, e:
	# Report exceptions to the coordinator.
		coord.request_stop(e)
	
	finally:
		# stop our queue threads and properly close the session
		print("This run can be identified as {}".format(run_identifier))
		coord.request_stop()
		coord.join(threads)
		sess.close()