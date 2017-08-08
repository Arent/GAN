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

# get training operations and placeholders
train_op_discrim, train_op_gen = create_training_operations()

# Merge all tensorboard summfaks
merged_summary_op = tf.summary.merge_all()
merged_summary_op = tf.identity(merged_summary_op, name="merged_summaries")

with tf.Session() as sess:
	saver = tf.train.Saver(max_to_keep=10)
	summary_writer = tf.summary.FileWriter("".join(['tensorboard_logs',"/",run_identifier]), tf.get_default_graph())


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

			_,  = sess.run([train_op_gen], feed_dict={Z:z_batch, real_images:image_batch})
			_,  = sess.run([train_op_discrim], feed_dict={Z:z_batch, real_images:image_batch})
			
			
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