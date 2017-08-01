import tensorflow as tf 
import numpy as np 

sess = tf.InteractiveSession()


def image_locations_and_labels_from_files(image_locations_filename, image_labels_filename):
	image_locations_file = open(image_locations_filename)
	image_labels_file = open(image_labels_filename)
	image_locations = image_locations_file.read().splitlines()
	image_labels = image_labels_file.read().splitlines()

	return image_locations, image_labels

def read_images_from_disk(file_queue):
	image_reader = tf.WholeFileReader()
	tf.Print(file_queue, [file_queue], message="This is a: ")


	value, label = image_reader.read(file_queue[0])
	images = tf.image.decode_jpeg(value, channels=3)

	return images, label




batch_size = 1
num_epochs = 10

image_locations, image_labels = image_locations_and_labels_from_files('data/image_locations','data/image_labels')
tf_image_locations = tf.convert_to_tensor(image_locations[0:10], dtype=tf.string)
tf_labels = tf.convert_to_tensor(image_labels[0:10], dtype=tf.string)
file_queue = tf.train.slice_input_producer([tf_image_locations, tf_labels],
											num_epochs=num_epochs,
											shuffle=True)
image, label = read_images_from_disk(file_queue)



sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord, start=True)

# Let's run the image, key tensors.
sess.run([image, label])

coord.request_stop()
coord.join(threads)





# 	coord.join(threads)