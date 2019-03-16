from param import parse_argument
from input_data import image_generator_xd
from models import FullModel
from models import AutoEncoder
from models import train_full_model
from models import check_loaded_weight
from models import CallbackParams 

# parse argument
args = parse_argument()

# extract useful variables from arguments
NCSVS = args.ncsvs
SIZE = args.size
BATCHSIZE = args.batchsize
BASESIZE = args.basesize
NCATS = args.n_class
IN_CHANNEL = args.in_channel
EPOCHS = args.n_epochs
VERBOSE = args.verbose
MODEL_PATH = args.model_path

# load autoencoder
autoencoder = AutoEncoder('/data/doodle/h5/autoencoder.h5', SIZE, IN_CHANNEL)
print('Successfully loaded autoencoder from disk')

# define full model
full_model_param = {
	'inputs': autoencoder.inputs,
	'autoencoder': autoencoder.model,
	'size': SIZE,
	'ncats': NCATS,
	'ae_trainable': True,
	'ngpu': args.n_gpu,
	'learn_rate': args.lrnrate,
}
full_model = FullModel(**full_model_param)
print('Successfully created full model')

# check if model weights are equal to each other
check_loaded_weight(full_model, autoencoder.model)

# load data generator
train_datagen = image_generator_xd(SIZE, BATCHSIZE, BASESIZE, NCATS, ks=range(NCSVS - 1))
validation_datagen = image_generator_xd(SIZE, BATCHSIZE, BASESIZE, NCATS, ks=range(NCSVS-1, NCSVS))
print('Successfully loaded data generator')

# start training
callback_params = CallbackParams(MODEL_PATH, 'val_top_3_accuracy')
hist, full_model = train_full_model(callback_params, full_model, train_datagen, validation_datagen, 5000, 50, EPOCHS, VERBOSE, './full_model')
