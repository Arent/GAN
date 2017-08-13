import tensorflow as tf 
from hyper_parameters import * 

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	mean = tf.reduce_mean(var)
	tf.summary.scalar('mean', mean)
	with tf.name_scope('stddev'):
	  stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	tf.summary.scalar('stddev', stddev)
	tf.summary.scalar('max', tf.reduce_max(var))
	tf.summary.scalar('min', tf.reduce_min(var))
	tf.summary.histogram('histogram', var)


def leaky_RELU(x):
	return tf.maximum(x, RELU_ALPHA * x)

def init_weight_variable(shape): #init_weight_variable generates a weight variable of a given shape.
	return tf.random_normal(shape, stddev=0.01)

def binary_cross_entropy(x , label):
	x = tf.clip_by_value(x, 1e-7, 1. - 1e-7) #for stability
	return -(label * tf.log(x) + (1.- label)*tf.log(1. - x))

def activate_convolution_transposed(input_dim, output_dim, strides, padding, input, activation , normalise=True):
	assert len(input_dim) == 3
	assert len(output_dim) == 3

	filter_shape = [output_dim[0], output_dim[1], output_dim[2], input_dim[2]]
	filter_weights = tf.get_variable("weights", initializer=init_weight_variable(filter_shape))

	batch_size = tf.shape(input)[0] 
	output_shape = [batch_size,output_dim[0],output_dim[1],output_dim[2]]

	output = tf.nn.conv2d_transpose(
					value=input,
					filter=filter_weights,
					output_shape=output_shape,
					strides=strides,
					padding=padding,
)
	bias = tf.get_variable("bias", initializer=init_weight_variable(output_dim))
	output_bias = output + bias

	if normalise:
		output_bias = tf.contrib.layers.batch_norm(output_bias, 
													center=True, 
													scale=True, 
													decay=NORMALISATION_DECAY)


	activated_output_bias= activation(output_bias)

	with tf.variable_scope("Activated_output_bias"):
		variable_summaries(activated_output_bias)
	with tf.variable_scope("Output_bias"):
		variable_summaries(output_bias)
	with tf.variable_scope("Weights"):
		variable_summaries(filter_weights)

	return activated_output_bias

def activate_convolution(filter_width, filter_height, input_dim, output_dim, strides, padding, input, activation, normalise=True):
	assert len(input_dim) == 3
	assert len(output_dim) == 3

	filter_shape = [filter_height, filter_width, input_dim[2], output_dim[2]]
	filter = tf.get_variable("weights", initializer=init_weight_variable(filter_shape))
	bias = tf.get_variable("bias", initializer=init_weight_variable(output_dim))


	output = tf.nn.conv2d(
				input=input,
				filter=filter,
				strides=strides,
				padding=padding)

	output_bias = output + bias
	if normalise:
		output_bias = tf.contrib.layers.batch_norm(output_bias, 
											center=True, 
											scale=True, 
											decay=NORMALISATION_DECAY)
	activated_output_bias= activation(output_bias)

	with tf.variable_scope("Activated_output_bias"):
		variable_summaries(activated_output_bias)
	with tf.variable_scope("Output_bias"):
		variable_summaries(output_bias)
	with tf.variable_scope("Weights"):
		variable_summaries(filter)
	return activated_output_bias

def activate_fully_connected(input, input_dim, output_dim, activation, normalize=True):
	weight_shape = [input_dim, output_dim[0]]
	weights = tf.get_variable(name="weights", initializer=init_weight_variable(weight_shape))
	bias = tf.get_variable(name="bias", initializer=init_weight_variable(output_dim))

	output = tf.matmul(input, weights)
  
	output_bias = output + bias
	if normalize:
		output_bias = tf.contrib.layers.batch_norm(output, 	
											center=True, 
											scale=True, 
											decay=NORMALISATION_DECAY)

	output_bias_logit = output_bias
	activated_output_bias = activation(output_bias)
	with tf.variable_scope("Activated_output_bias"):
		variable_summaries(activated_output_bias)
	with tf.variable_scope("Output_bias"):
		variable_summaries(output_bias)
	with tf.variable_scope("Weights"):
		variable_summaries(weights)
	
	return activated_output_bias, output_bias_logit

def generate(z):
	batch_size = tf.shape(z)[0] 

	with tf.variable_scope("generator"):
		z = tf.reshape(z,[batch_size, 1, 1, 100])
		with tf.variable_scope("layer1"):
			activated_layer_1 = activate_convolution_transposed(\
											input_dim=[1,1,100],output_dim=[4,4,1024], 
											strides=[1,1,1,1], padding='VALID', 
											input=z,
											activation=tf.nn.relu,
											normalise=True) 

		with tf.variable_scope("layer2"):
			activated_layer_2 = activate_convolution_transposed(\
											input_dim=[4,4,1024],output_dim=[8,8,512], 
											strides=[1,2,2,1], padding='SAME', 
											input=activated_layer_1,
											activation=tf.nn.relu,
											normalise=True)

		with tf.variable_scope("layer3"):
			activated_layer_3 = activate_convolution_transposed(\
											input_dim=[8,8,512],output_dim=[16,16,256], 
											strides=[1,2,2,1], padding='SAME', 
											input=activated_layer_2,
											activation=tf.nn.relu,
											normalise=True)

		with tf.variable_scope("layer4"):
			activated_layer_4 = activate_convolution_transposed(\
											input_dim=[16,16,256],output_dim=[32,32,128], 
											strides=[1,2,2,1], padding='SAME', 
											input=activated_layer_3,
											activation=tf.nn.relu,
											normalise=True)

		with tf.variable_scope("layer5"):
			activated_layer_5 = activate_convolution_transposed(\
											input_dim=[32,32,128],output_dim=[64,64,3], 
											strides=[1,2,2,1], padding='SAME', 
											input=activated_layer_4,
											activation=tf.nn.relu,
											normalise=True)
		fake_image = activated_layer_5 		
	return fake_image

def discriminate(image):
	batch_size = tf.shape(image)[0] 
	with tf.variable_scope("discriminator"):
		with tf.variable_scope("layer1"):
			activated_layer_1 = activate_convolution(\
											filter_width=KERNEL_WIDTH, filter_height=KERNEL_HEIGHT,
											input_dim=[64, 64, 3], output_dim=[32, 32, 128], 
											strides=[1,2,2,1], padding='SAME', 
											input=image,
											activation=leaky_RELU,
											normalise=True) 

		with tf.variable_scope("layer2"):
			activated_layer_2 = activate_convolution(\
											filter_width=KERNEL_WIDTH, filter_height=KERNEL_HEIGHT,
											input_dim=[32, 32, 128], output_dim=[16, 16, 256], 
											strides=[1,2,2,1], padding='SAME', 
											input=activated_layer_1,
											activation=leaky_RELU,
											normalise=True)

		with tf.variable_scope("layer3"):
			activated_layer_3 = activate_convolution(\
											filter_width=KERNEL_WIDTH, filter_height=KERNEL_HEIGHT,
											input_dim=[16, 16, 256], output_dim=[8, 8, 512], 
											strides=[1,2,2,1], padding='SAME', 
											input=activated_layer_2,
											activation=leaky_RELU,
											normalise=True) 

		with tf.variable_scope("layer4"):
			activated_layer_4 = activate_convolution(\
											filter_width=KERNEL_WIDTH, filter_height=KERNEL_HEIGHT,
											input_dim=[8, 8, 512], output_dim=[4, 4, 1024], 
											strides=[1,2,2,1], padding='SAME', 
											input=activated_layer_3,
											activation=leaky_RELU,
											normalise=True) 

		with tf.variable_scope("layer5"):
			total_dimension =  tf.reduce_prod(tf.shape(activated_layer_4)[1:])
			activated_layer_4_flattened = tf.reshape(activated_layer_4, [batch_size,  total_dimension])#4*4*1024])#
			judgement, logit_judgement = activate_fully_connected(
											input=activated_layer_4_flattened,
											input_dim=4*4*1024, 
											output_dim=[1],
											normalize=True,
											activation=tf.nn.tanh) 
	return judgement, logit_judgement

def create_loss_functions(probability_real, logit_real,probability_fake, logit_fake):
	#Create loss functions 
	loss_discriminator_real = tf.reduce_mean(binary_cross_entropy(	x=probability_real, 
												label=tf.ones_like(probability_real)))
	loss_discriminator_fake = tf.reduce_mean(binary_cross_entropy(	x=probability_fake, 
												label=tf.zeros_like(probability_real)))
	loss_discriminator 	= loss_discriminator_real + loss_discriminator_fake

	loss_generator = tf.reduce_mean(binary_cross_entropy(x=probability_fake,
										label=tf.ones_like(probability_fake)))

	# summarize loss functions
	tf.summary.scalar("discriminator_real", tf.reduce_mean(loss_discriminator_real))
	tf.summary.scalar("discriminator", loss_discriminator)
	tf.summary.scalar("generator", loss_generator)
	tf.summary.scalar("discriminator_fake", tf.reduce_mean(loss_discriminator_fake))		

	return  loss_discriminator, loss_generator

def build_graph():
	Z = tf.placeholder(dtype=tf.float32, shape=(None, Z_DIMENSION), name='Z')
	real_images = tf.placeholder(	dtype=tf.float32, 
									shape=(None,IMAGE_HEIGHT,IMAGE_WIDTH,NUM_CHANNELS),
									name='real_images')


	with tf.variable_scope("model") as scope:
		fake_images = generate(Z)
		tf.identity(fake_images, "fake_images") 
		probability_real, logit_real = discriminate(real_images)
		scope.reuse_variables() #Ensure the discriminate functions doesn't create new variables
		probability_fake, logit_fake = discriminate(fake_images)

		loss_discriminator, loss_generator =create_loss_functions(probability_real, 
												logit_real,	probability_fake, logit_fake)	
		tf.identity(loss_discriminator, "loss_discriminator") 
		tf.identity(loss_discriminator, "loss_generator") 




def create_training_operations():
	# retrieve the loss functions and variable list
	# with tf.variable_scope("model") as scope:

	loss_discriminator = tf.get_default_graph()\
							.get_tensor_by_name("model/loss_discriminator:0")

	loss_generator = tf.get_default_graph()\
							.get_tensor_by_name("model/loss_generator:0")
	
	discriminator_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
								scope='model/discriminator')
	generator_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
								scope='model/generator')

	#Create optimizer 
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_ADAM)
	
	#Explicitly create gradients for optimizing and logging
	discriminator_gradients = optimizer.compute_gradients(loss_discriminator, var_list=discriminator_variables)
	generator_gradients = optimizer.compute_gradients(loss_generator, var_list=generator_variables)

	# Summarize all gradients
	for grad, var in discriminator_gradients + generator_gradients:
		if not grad == None:
			tf.summary.histogram(var.name + '/gradient', grad)

	#Create and identiy the training operations
	train_op_discrim = optimizer.apply_gradients(grads_and_vars=discriminator_gradients, 
		name='train_operation_discriminator')
	train_op_gen	 = optimizer.apply_gradients(grads_and_vars=generator_gradients, 
		name='train_operation_generator')

	# Merge all tensorboard summaries, identidy the merged operations
	merged_summary_op = tf.summary.merge_all()
	tf.identity(merged_summary_op, name="merged_summaries")
	





