import numpy as np
import tensorflow as tf
from digit_recognizer_input import *
from digit_recognizer_model import *

num_labels = 10
learning_rate = 0.05
batch_size = 128
num_steps = 3001

def reformat(dataset, labels):

	dataset = dataset.astype(np.float32)
	labels = (labels[:, None].astype(int) == np.arange(num_labels)).astype(np.float32)

	return dataset, labels

def run_training():

	with tf.Graph().as_default():
		# Creating placeholder for images and labels
		images_placeholder, labels_placeholder = placeholder_input()

		# Builds a graph that computes inference		
		logits = inference(images_placeholder)

		# Adding loss op to the graph
		loss = loss_op(logits, labels_placeholder)

		# Adding train op to the graph
		train = train_op(loss, learning_rate)

		# Adding accuracy op to the graph
		score = accuracy(logits, labels_placeholder)

		with tf.Session() as sess:
			# Initializing all variables
			init = tf.initialize_all_variables()
			sess.run(init)
			print 'Graph Initialized'

			# Loading dataset
			print 'Loading dataset'
			dataset = load_digits()
			print 'Dataset loaded'
			train_dataset = dataset['train_dataset']
			validation_dataset = dataset['validation_dataset']

			# Reshaping images and labels for training and validation dataset
			train_images, train_labels = reformat(train_dataset.data, train_dataset.target)
			validation_data, validation_labels = reformat(validation_dataset.data, validation_dataset.target)

			validation_feed_dict = {images_placeholder: validation_data, labels_placeholder: validation_labels}

			for step in xrange(num_steps):
				offset = (step * batch_size) % (train_images.shape[0] - batch_size)

				batch_data = train_images[offset:(offset + batch_size), :]
				batch_labels = train_labels[offset:(offset + batch_size), :]

				feed_dict = {images_placeholder: batch_data, labels_placeholder: batch_labels}

				l, _, train_accuracy = sess.run([loss, train, score], feed_dict=feed_dict)

				if step % 500 == 0:
					print 'Minibatch loss at step %d: %f' % (step, l)
					print '  Training Accuracy: %.3f' % train_accuracy
					print '  Validation Accuracy: %.3f' % sess.run(score, feed_dict=validation_feed_dict)

if __name__ == '__main__':
	run_training()		

