import argparse

def parse_argument():
	parser = argparse.ArgumentParser(description='Run sequence analysis')
	parser.add_argument('--n_class', help='number of label', type=int, default=340)
	parser.add_argument('--n_gpu', help='number of label', type=int, default=1)
	parser.add_argument('--n_epochs', help='number of label', type=int, default=16)
	parser.add_argument('--verbose', help='if 1 it prints training steps, otherwise 0', type=int, default=1)
	parser.add_argument('--lrnrate', help='number of label', type=float, default=0.005)
	parser.add_argument('--batchsize', help='number of data per batch', type=int, default=680)
	parser.add_argument('--basesize', help='number of data per batch', type=int, default=256)
	parser.add_argument('--size', help='number of iteration for learning', type=int, default=64)
	parser.add_argument('--ncsvs', help='number of iteration for learning', type=int, default=100)
	parser.add_argument('--in_channel', help='number of iteration for learning', type=int, default=1)
	parser.add_argument('--ct_dir', help='category name csv folder', default='/data/doodle/train')
	parser.add_argument('--dp_dir', help='data directory', default='/data/doodle/train_sampled')
	parser.add_argument('--model_path', help='data directory', default='h5/full_model.h5')


	args = parser.parse_args()
	return args
