import tensorflow as tf 
from hyper_parameters import * 

#generator
# 100 x 1 -> 4x4 X 1024 
def leaky_RELU(x):
	return tf.maximum(x, relu_alpha * x)

def transposed_convolution_relu_normalisation(input_dim, output_dim, strides, padding, input, normalise=True, activation):
	assert input_dim[1] < output_dim[1] 
	assert input_dim[2] < output_dim[2] 
	assert len(input_dim) == 3
	assert len(output_dim) == 3

	filter_shape = [output_dim[0], output_dim[1], output_dim[2], input_dim[2]]
	filter_weights = tf.get_variable("weights", filter_shape)

	batch_size = tf.shape(input)[0] 
	output_shape = [batch_size,output_dim[0],output_dim[1],output_dim[2]]

	unactivated_output = tf.nn.conv2d_transpose(
    value=input,
    filter=filter_weights,
    output_shape=output_shape,
    strides=strides,
    padding=padding,
)
	bias = tf.get_variable("bias", output_dim)
 	unactivated_output_bias = unactivated_output + bias

 	if normalise:
    	unactivated_output_bias = tf.contrib.layers.batch_norm(unactivated_output_bias, 
    									center=True, 
    									scale=True, 
    									decay=normalisation_decay)


 	activated_output_bias= activation(unactivated_output_bias)

	return activated_output_bias

def generator(z):
	with tf.variable_scope("generator"):
		with tf.variable_scope("layer1"):
			generator_activated_layer_1 = transposed_convolution_relu_normalisation(\
				input_dim=[1,1,100],output_dim=[4,4,1024], 
				strides=[1,1,1,1], padding='SAME', input=z,
				normalise=True, activation=leaky_RELU)

		with tf.variable_scope("layer2"):
			generator_activated_layer_2 = transposed_convolution_relu_normalisation(\
				input_dim=[4,4,1024],output_dim=[8,8,512], 
				strides=[1,2,2,1], padding='SAME', input=generator_activated_layer_1,
				normalise=True, activation=leaky_RELU)

		with tf.variable_scope("layer3"):
			generator_activated_layer_3 = transposed_convolution_relu_normalisation(\
				input_dim=[8,8,512],output_dim=[16,16,256], 
				strides=[1,2,2,1], padding='SAME', input=generator_activated_layer_2,
				normalise=True, activation=leaky_RELU)

		with tf.variable_scope("layer4"):
			generator_activated_layer_4 = transposed_convolution_relu_normalisation(\
				input_dim=[16,16,256],output_dim=[32,32,128], 
				strides=[1,2,2,1], padding='SAME', input=generator_activated_layer_3,
				normalise=True, activation=leaky_RELU)


		with tf.variable_scope("layer5"):
			generator_activated_layer_5 = transposed_convolution_relu_normalisation(\
				input_dim=[32,32,128],output_dim=[64,64,3], 
				strides=[1,2,2,1], padding='SAME', input=generator_activated_layer_4,
				normalise=True, activation=leaky_RELU)

		generated_image = generator_activated_layer_5			
	return generated_image