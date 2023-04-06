import argparse

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type = str, default = '../dataset/',
				   help='path to data')
	parser.add_argument('--model_path', type = str, default = '../model_save/',
				   help='path to save model')
	parser.add_argument('--name', type = str, default = '',
				   help='path to save model')

	parser.add_argument('--D_n', type = int, default = 1306529,
				   help = 'number of url node')
	
	parser.add_argument('--in_f_d', type = int, default = 768,
				   help = 'input feature dimension(domain embedding & node net embedding)')
	parser.add_argument('--embed_d', type = int, default = 512,
				   help = 'embedding dimension')
	parser.add_argument('--lr', type = int, default = 0.001,
				   help = 'learning rate')
	parser.add_argument('--batch_s', type = int, default = 20000,
				   help = 'batch size')
	parser.add_argument('--mini_batch_s', type = int, default = 128,
				   help = 'mini batch size')
	parser.add_argument('--epoch', type = int, default = 200,
				   help = 'number of epochs')
	parser.add_argument('--p1', type = int, default = 0.2,
				   help='')
	parser.add_argument('--p2', type = int, default = 0.2,
				   help='')
	parser.add_argument('--p3', type=int, default=0.2,
						help='')
	parser.add_argument('--p4', type = int, default = 0.2,
				   help='')
	parser.add_argument('--p5', type = int, default = 0.2,
				   help='')
	parser.add_argument("--random_seed", default = 10, type = int)
	parser.add_argument('--train_test_label', type= int, default = 0,
				   help='train/test label: 0 - train, 1 - test, 2 - code test/generate negative ids for evaluation')
	parser.add_argument('--save_model_freq', type = float, default = 2,
				   help = 'number of iterations to save model')
	parser.add_argument("--cuda", default = 0, type = int)
	parser.add_argument("--checkpoint", default = 'sample13_p02p02_20000395.pt' , type=str)
	args = parser.parse_args()

	return args
