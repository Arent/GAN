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
	assert os.path.isdir("".join([model_folder,run_identifier])) == False
	os.mkdir("".join([model_folder,run_identifier]))
	os.mkdir("".join([model_folder,run_identifier,"/",sample_folder]))
	return "".join([model_folder,run_identifier,"/"])
save_location = create_save_location()

def rescale_and_save_image(image_array, iBatch):
	image_array = np.uint8((image_array +1)*127.5) # rescale and make into integer\
	image = Image.fromarray(image_array)
	image.save("".join([model_folder,run_identifier,"/",sample_folder,str(iBatch),".jpeg"]))

#Get que with training files
n_train_files, n_test_files\
,train_image_batch, train_label_batch, test_image_batch, test_label_batch = get_batches()
batches_per_epoch = int(math.ceil(float(n_train_files) / BATCH_SIZE))

#Launch graph
with tf.Session() as sess:
	#get training operations and placeholders
	build_graph()
	graph = tf.get_default_graph()

	create_training_operations()
	train_op_discrim = graph.get_operation_by_name("train_operation_discriminator")
	train_op_gen = graph.get_operation_by_name("train_operation_generator")
	merged_summary_op = graph.get_operation_by_name("merged_summaries")
	images_tensor = graph.get_tensor_by_name('model/fake_images:0')

	saver = tf.train.Saver(tf.trainable_variables(),max_to_keep=None)
	summary_writer = tf.summary.FileWriter("".join(['tensorboard_logs',"/",run_identifier]), tf.get_default_graph())

	Z = tf.get_default_graph().get_tensor_by_name('Z:0')
	real_images =tf.get_default_graph().get_tensor_by_name('real_images:0') 
	
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

			#retrieve and display epoch and batch
			epoch = int(iBatch / batches_per_epoch) + 1
			batch_number = iBatch % batches_per_epoch
			print('Epoch: '+str(epoch) +'/'+str(EPOCHS) + ' Batch: ' + str(batch_number) +'/' + str(batches_per_epoch -1))
			
			#Get Z values and image batch
			z_batch = np.random.uniform(-1, 1, size=[BATCH_SIZE, Z_DIMENSION]).astype(np.float32)
			image_batch  = sess.run(train_image_batch)

			#Do the acual training
			_,  = sess.run([train_op_gen], feed_dict={Z:z_batch, real_images:image_batch})
			_,  = sess.run([train_op_discrim], feed_dict={Z:z_batch, real_images:image_batch})
			
			#Add summaries of variables to tensorboard
			summary_str = sess.run(merged_summary_op, feed_dict={Z:z_batch, real_images:image_batch})
			summary_writer.add_summary(summary_str, iBatch)
			if iBatch == 0: #Save metagraph only once (verrrrrryy large) 
				saved_path = saver.save(sess, save_location  +str(batch_number), write_meta_graph=True)
			
			if batch_number % 20 == 0: #save variable state and make sample image
				images = sess.run(images_tensor,feed_dict={Z:z_batch})
				rescale_and_save_image(images[0,:,:,:], iBatch)
				saved_path = saver.save(sess, save_location +"Epoch_"+str(epoch) + "_Batch_" +str(batch_number)
									,write_meta_graph=False)
				print("Model saved in file: %s" % saved_path)
			iBatch = iBatch + 1
	
	except Exception, e:
	# Report exceptions to the coordinator.
		coord.request_stop(e)
	
	finally:
		# stop our queue threads and properly close the session
		print("This run can be identified as {}".format(run_identifier))
		print("Run the command line:\n" \
		"--> tensorboard --logdir=/tmp/tensorflow_logs " \
		"\nThen open http://0.0.0.0:6006/ into your web browser")

		coord.request_stop()
		coord.join(threads)
		sess.close()