import numpy as np
from functools import reduce
from tensorflow import keras
from keras.applications.mobilenet import MobileNet
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Input, Reshape, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import top_k_categorical_accuracy, categorical_crossentropy, categorical_accuracy
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras.utils.training_utils import multi_gpu_model

class CallbackParams(object):
	def __init__(self, full_path, monitor_func):
		self.reduce_param = {
			'monitor':monitor_func,
			'factor':0.75,
			'patience':3,
			'min_delta':0.001,
			'mode':'max',
			'min_lr':1e-5,
			'verbose':1,
		}
		self.checkpoint_param = {
			'filepath':full_path,
			'monitor':monitor_func,
			'mode':'max',
			'save_best_only':True,
			'save_weights_only':True,
		}

def top_3_accuracy(y_true, y_pred):
	return top_k_categorical_accuracy(y_true, y_pred, k=3)

def encoder(inputs):
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) #28 x 28 x 32
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #7 x 7 x 128 (small and thick)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	return conv4

def decoder(conv4):
	conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)
	conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5) # 14 x 14 x 32
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)
	up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
	return decoded

class AutoEncoder(object):
	def __init__(self, model_name, size, in_channel, use_pretrained=True):
		self.inputs = Input(shape = (size, size, in_channel))
		self.model = Model(self.inputs, decoder(encoder(self.inputs)))
		if use_pretrained:
			self.model.load_weights(model_name)
		self.size = size
		
def check_loaded_weight(full_model, autoencoder):
	tmp_list = full_model.get_weights()[0][1] == autoencoder.get_weights()[0][1]
	tmp_list = tmp_list.flatten()
	mul_numbers = reduce(lambda x ,y : x*y , tmp_list)
	if mul_numbers:
		print('loaded weight is exactly equalt to each other')
	else:
		print('loaded weight is not exactly equalt to each other')

def fc(enco, size, ncats):
	mobile = MobileNet(input_shape=(size, size, 128), alpha=1., weights=None, classes=ncats)(enco)
	return mobile

def FullModel(inputs,
			autoencoder,
			size, 
			ncats, 
			ngpu=1, 
			learn_rate=0.005, 
			use_pretrained_model=False,
			use_pretrained_ae=True,
			ae_trainable=False):
	encoding = encoder(inputs)
	full_model = Model(inputs, fc(encoding, size, ncats))

	if use_pretrained_model:
		full_model.load_weights('/data/doodle/h5/full_model.h5')

	else:
		if use_pretrained_ae:
			# set autoencoder weight
			for l1,l2 in zip(full_model.layers, autoencoder.layers):
				try:
					l1.set_weights(l2.get_weights())
					l1.trainable = ae_trainable
					print('updated weight of %s' % l1)
				except ValueError:
					continue

	return full_model

def train_full_model(
					callback_params,
					full_model, 
					train_datagen,
					validation_datagen, 
					train_steps_per_epoch, 
					val_steps_per_epoch,
					epochs,
					verbose,
					full_model_path=None):
	callbacks = [
		ReduceLROnPlateau(**(callback_params.reduce_param)),
		ModelCheckpoint(**(callback_params.checkpoint_param)),
	]

	hist = full_model.fit_generator(
		train_datagen,
		steps_per_epoch=train_steps_per_epoch,
		epochs=epochs,
		verbose=verbose,
		validation_data=validation_datagen,
		validation_steps=val_steps_per_epoch,
		callbacks = callbacks
	)

	return hist, full_model
