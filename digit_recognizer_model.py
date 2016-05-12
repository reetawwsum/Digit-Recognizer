import numpy as np
import tensorflow as tf

image_size = 28
num_labels = 10

def placeholder_input():
	images_placeholder = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
	labels_placeholder = tf.placeholder(tf.float32, shape=(None, num_labels))

	return images_placeholder, labels_placeholder

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def inference(images):
	# Hidden 1
	with tf.name_scope('hidden1'):
		weights = weight_variable([784, 2500])
		biases = bias_variable([2500])

		hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

	# Hidden 2
	with tf.name_scope('hidden2'):
		weights = weight_variable([2500, 2000])
		biases = bias_variable([2000])

		hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

	# Hidden 3
	with tf.name_scope('hidden3'):
		weights = weight_variable([2000, 1500])
		biases = bias_variable([1500])

		hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)

	# Hidden 4
	with tf.name_scope('hidden4'):
		weights = weight_variable([1500, 1000])
		biases = bias_variable([1000])

		hidden4 = tf.nn.relu(tf.matmul(hidden3, weights) + biases)

	# Hidden 5
	with tf.name_scope('hidden5'):
		weights = weight_variable([1000, 500])
		biases = bias_variable([500])

		hidden5 = tf.nn.relu(tf.matmul(hidden4, weights) + biases)

	# Linear Layer
	with tf.name_scope('linear'):
		weights = weight_variable([500, 10])
		biases = bias_variable([10])

		logits = tf.matmul(hidden5, weights) + biases

	return logits

def loss_op(logits, labels):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

	return loss	

def train_op(loss, learning_rate):
	optimizer = tf.train.AdagradOptimizer(learning_rate)
	train = optimizer.minimize(loss)

	return train

def accuracy(predictions, labels):
	correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))

	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
	pass

