import tensorflow as tf 
from hyper_parameters import * 


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

	print(input.get_shape())
	filter_shape = [output_dim[0], output_dim[1], output_dim[2], input_dim[2]]
	print('filter shape' + str(filter_shape))
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
	print('fake image dimensions: ' + str(fake_image.get_shape()))
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
			print('activated_layer_4 shape' + str(activated_layer_4.get_shape()))
			total_dimension =  tf.reduce_prod(tf.shape(activated_layer_4)[1:])
			activated_layer_4_flattened = tf.reshape(activated_layer_4, [batch_size,  total_dimension])#4*4*1024])#
			print('activated_layer_4_flattened shape' + str(activated_layer_4_flattened.get_shape()))
			judgement, logit_judgement = activate_fully_connected(
											input=activated_layer_4_flattened,
											input_dim=4*4*1024, 
											output_dim=[1],
											normalize=True,
											activation=tf.nn.tanh) 
	return judgement, logit_judgement

def build_model():
	with tf.variable_scope("model") as scope:
		Z = tf.placeholder(dtype=tf.float32, shape=(None, Z_DIMENSION))
		real_images = tf.placeholder(	dtype=tf.float32, 
										shape=(None,IMAGE_HEIGHT,IMAGE_WIDTH,NUM_CHANNELS))
		fake_images = generate(Z)


		probability_real, logit_real = discriminate(real_images)
		scope.reuse_variables() #Ensure the discriminate functions doesn't create new variables
		probability_fake, logit_fake = discriminate(fake_images)

		loss_discriminator_real = binary_cross_entropy(	x=probability_real, 
														label=tf.ones_like(probability_real))
		loss_discriminator_fake = binary_cross_entropy(	x=probability_fake, 
														label=tf.zeros_like(probability_real))
		loss_discriminator 	= tf.reduce_mean(loss_discriminator_real) \
							+ tf.reduce_mean(loss_discriminator_fake)
		loss_generator = tf.reduce_mean(binary_cross_entropy(	x=probability_fake,
															label=tf.ones_like(probability_fake)))
	return Z, real_images, probability_real, logit_real, probability_fake, logit_fake,\
			loss_discriminator_real, loss_discriminator_fake, loss_discriminator ,loss_generator 

