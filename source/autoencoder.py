import tensorflow as tf

class AUTOENCODER(object):
	def __init__(self, args, n_hid_1, n_hid_2, size):
		self.n_vis = size*size
		self.n_hid_1 = n_hid_1
		self.n_hid_2 = n_hid_2
		self.size = size
		self.data = tf.placeholder(tf.float64, [None, size, size, 1])
		self.input_x = tf.reshape(self.data, [-1, size*size])
		self.input_x = tf.cast(self.input_x, tf.float32)

		# run decoder and decoder
		self.encoded = self.encoder()
		self.decoded = self.decoder()

		# define loss
		self.loss = tf.losses.mean_squared_error(self.input_x, self.decoded)
		self.train_op = tf.train.RMSPropOptimizer(0.05).minimize(self.loss)

	def encoder(self):
		encoder_weight_1 = tf.Variable(\
			tf.random_uniform([self.n_vis, self.n_hid_1], -0.1, 0.1), name='enc_weight_1')
		encoder_bias_1 = tf.Variable(tf.constant(0.1, shape=[self.n_hid_1]), name='dec_bias_1')

		encoder_weight_2 = tf.Variable(\
			tf.random_uniform([self.n_hid_1, self.n_hid_2], -0.1, 0.1), name='enc_weight_2')
		encoder_bias_2 = tf.Variable(tf.constant(0.1, shape=[self.n_hid_2]), name='enc_bias_2')

		lay1_encoded = tf.nn.sigmoid(tf.matmul(self.input_x, encoder_weight_1) + encoder_bias_1)
		lay2_encoded = tf.nn.sigmoid(tf.matmul(lay1_encoded, encoder_weight_2) + encoder_bias_2)
		return lay2_encoded

	def decoder(self):
		decoder_weight_1 = tf.Variable(\
			tf.random_uniform([self.n_hid_2, self.n_hid_1], -0.1, 0.1), name='dec_weight_1')
		decoder_bias_1 = tf.Variable(tf.constant(0.1, shape=[self.n_hid_1]), name='dec_bias_1')

		decoder_weight_2 = tf.Variable(\
			tf.random_uniform([self.n_hid_1, self.n_vis], -0.1, 0.1), name='dec_weight_2')
		decoder_bias_2 = tf.Variable(tf.constant(0.1, shape=[self.n_vis]), name='dec_bias_2')

		lay1_decoded = tf.nn.sigmoid(tf.matmul(self.encoded, decoder_weight_1) + decoder_bias_1)
		lay2_decoded = tf.nn.sigmoid(tf.matmul(lay1_decoded, decoder_weight_2) + decoder_bias_2)
		return lay2_decoded

