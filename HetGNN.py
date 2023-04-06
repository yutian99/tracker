import torch
import torch.optim as optim
import data_generator
import tools
from args import read_args
from torch.autograd import Variable
import numpy as np
import random
import logging
import re
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
logging.basicConfig(level=logging.INFO,format="%(asctime)s-%(name)s-%(levelname)s-%(message)s")
torch.set_num_threads(2)
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


class model_class(object):
	def __init__(self, args):
		super(model_class, self).__init__()
		self.args = args
		self.gpu = args.cuda

		input_data = data_generator.input_data(args = self.args)

		self.input_data = input_data

		feature_list = [input_data.d_graph_embed, input_data.d_name_embed]
		logging.info("tracker num:{}".format(input_data.tracker))

		for i in range(len(feature_list)):
			feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()

		if self.gpu>=0:
			for i in range(len(feature_list)):
				feature_list[i] = feature_list[i].cuda()

		tracker_idx=input_data.tracker_idx
		normal_idx=input_data.normal_idx
		random.shuffle(normal_idx)
		data = normal_idx[:len(tracker_idx)*3]+tracker_idx
		#data = [i for i in range(self.args.D_n)]
		random.shuffle(data)
		self.data=data
		train_n = int(len(data) * 0.8)
		batch_n = int(train_n / self.args.mini_batch_s)
		self.batch_n=batch_n
		self.train_data=data[:train_n]
		self.test_data=data[train_n:]
		logging.info("train num:{},test num:{}".format(len(self.train_data),len(self.test_data)))

		self.model = tools.HetAgg(args, feature_list)

		if self.gpu>=0:
			self.model.cuda()
		self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay = 0)
		self.model.init_weights()

	def get_f1(self,predict, label):
		TP = 0
		TN = 0
		FN = 0
		FP = 0
		precision = 0
		recall = 0
		N = len(predict)
		for i in range(N):
			# print(predict[i],label[i])
			if predict[i] == 1 and label[i] == 1:
				TP += 1
			if predict[i] == 0 and label[i] == 0:
				TN += 1
			if predict[i] == 0 and label[i] == 1:
				FN += 1
			if predict[i] == 1 and label[i] == 0:
				FP += 1
		print(TP, TN, FP, FN)
		if TP + FP != 0:
			precision = TP / (TP + FP)
		if TP + FN != 0:
			recall = TP / (TP + FN)
		accuracy = (TP + TN) / (TP + FP + TN + FP)
		F1 = 2 * recall * precision / (recall + precision)
		return accuracy, precision, recall, F1
	def model_test(self):
		if self.args.checkpoint != '':
			self.model.load_state_dict(torch.load(self.args.model_path+self.args.checkpoint))
			logging.info("model loaded...")
		data=self.test_data
		TP = 0
		TN = 0
		FN = 0
		FP = 0
		loss = 0
		embed_d = self.args.embed_d
		self.model.eval()
		random.shuffle(data)
		mini_batch_s = self.args.mini_batch_s
		mini_batches = [data[k:k + mini_batch_s] for k in range(0, len(data), mini_batch_s)]

		for mini_batch in mini_batches:
			c_out= self.model(mini_batch, embed_d)
			loss_tmp,TP_tmp,TN_tmp,FN_tmp,FP_tmp = tools.cross_entropy_loss(c_out, mini_batch,self.input_data.d_labels)
			TP+=TP_tmp
			TN += TN_tmp
			FN += FN_tmp
			FP += FP_tmp
			loss+=loss_tmp
		precision=0
		recall=0
		F1=0
		logging.info('\nTP:{}\tTN:{}\tFN:{}\tFP:{}'.format(TP, TN, FN, FP))
		if TP + FP != 0:
			precision = TP / (TP + FP)
		if TP + FN != 0:
			recall = TP / (TP + FN)
		if recall+precision != 0:
			F1=2*recall*precision/(recall+precision)
		acc = (TP + TN) / (TP + TN + FP + FN)
		logging.info('\nTest set:precision:{:.6f}\trecall:{:.6f}\tF1:{:.6f}\tacc:{:.6f}\t\tLoss: {:.6f}\n'
			  .format( precision, recall, F1, acc, loss/len(mini_batches)))


	def model_train(self):
		logging.info('model training ...')
		if self.args.checkpoint != '':
			self.model.load_state_dict(torch.load(self.args.model_path+self.args.checkpoint))
			logging.info("model loaded...")
		
		self.model.train()
		mini_batch_s = self.args.mini_batch_s
		embed_d = self.args.embed_d
		data=self.train_data


		for iter_i in range(self.args.epoch):
			TP = 0
			TN = 0
			FN = 0
			FP = 0
			loss=0
			random.shuffle(data)
			mini_batches = [data[k:k + mini_batch_s] for k in range(0, len(data), mini_batch_s)]


			total = 0
			F1 = 0
			recall=0
			precision=0
			for mini_batch in mini_batches:
				c_out= self.model(mini_batch, embed_d)
				loss_tmp,TP_tmp,TN_tmp,FN_tmp,FP_tmp = tools.cross_entropy_loss(c_out, mini_batch,self.input_data.d_labels)
				#correct+=correct_mini
				total+=len(mini_batch)
				TP+=TP_tmp
				TN += TN_tmp
				FN += FN_tmp
				FP += FP_tmp
				loss+=loss_tmp
				self.optim.zero_grad()
				loss_tmp.backward()
				self.optim.step()
			if TP+FP!=0:
				precision = TP / (TP + FP)
			if TP + FN != 0:
				recall = TP / (TP + FN)
			acc = (TP + TN) / (TP + TN + FP + FN)
			if recall + precision != 0:
				F1 = 2*recall*precision / (recall + precision)

			if iter_i%1==0:
				logging.info('\nTrain Epoch{}:precision:{:.6f}\trecall:{:.6f}\tF1:{:.6f}\tacc:{:.6f}\t\tLoss: {:.6f}'
						.format(iter_i, precision,recall,F1,acc, loss/len(mini_batches)))
				logging.info('\nTP:{}\tTN:{}\tFN:{}\tFP:{}'.format(TP, TN, FN, FP))

				logging.info('epoch ' + str(iter_i) + ' finish.\n')
			if iter_i%50==0:
				torch.save(self.model.state_dict(), self.args.model_path + self.args.name + str(iter_i+0) + ".pt")
				#self.model_test()
				#self.model.train()
	def DT(self):
		feature_dim = 768
		X = np.zeros((len(self.data), feature_dim))
		y = np.zeros(len(self.data))
		data=self.data
		for i in range(len(self.data)):
			X[i]=self.input_data.d_name_embed[data[i]]
			y[i]=self.input_data.d_labels[data[i]]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
		clf = tree.DecisionTreeClassifier(criterion='gini')
		clf.fit(X_train, y_train)
		print("train:")
		predict_train = clf.predict(X_train)
		accuracy, precision, recall, F1 = self.get_f1(predict_train, y_train)
		print('precision:{:.6f}\trecall:{:.6f}\tF1:{:.6f}\tacc:{:.6f}'
			  .format(precision, recall, F1, accuracy))
		print("test:")
		predict_test = clf.predict(X_test)
		accuracy, precision, recall, F1 = self.get_f1(predict_test, y_test)
		print('precision:{:.6f}\trecall:{:.6f}\tF1:{:.6f}\tacc:{:.6f}'
			  .format(precision, recall, F1, accuracy))
	def RF(self):
		feature_dim = 768
		X = np.zeros((len(self.data), feature_dim))
		y = np.zeros(len(self.data))
		data=self.data
		for i in range(len(self.data)):
			X[i]=self.input_data.d_name_embed[data[i]]
			y[i]=self.input_data.d_labels[data[i]]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
		forest = RandomForestClassifier()
		forest.fit(X_train, y_train)
		print("train:")
		predict_train = forest.predict(X_train)
		accuracy, precision, recall, F1 = self.get_f1(predict_train, y_train)
		print('precision:{:.6f}\trecall:{:.6f}\tF1:{:.6f}\tacc:{:.6f}'
			  .format(precision, recall, F1, accuracy))
		print("test:")
		predict_test = forest.predict(X_test)
		accuracy, precision, recall, F1 = self.get_f1(predict_test, y_test)
		print('precision:{:.6f}\trecall:{:.6f}\tF1:{:.6f}\tacc:{:.6f}'
			  .format(precision, recall, F1, accuracy))
	def LR(self):
		feature_dim = 768
		X = np.zeros((len(self.data), feature_dim))
		y = np.zeros(len(self.data))
		data = self.data
		for i in range(len(self.data)):
			X[i] = self.input_data.d_name_embed[data[i]]
			y[i] = self.input_data.d_labels[data[i]]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
		LRmodel = LogisticRegression()
		LRmodel.fit(X_train, y_train)
		print("train:")
		predict_train = LRmodel.predict(X_train)
		accuracy, precision, recall, F1 = self.get_f1(predict_train, y_train)
		print('precision:{:.6f}\trecall:{:.6f}\tF1:{:.6f}\tacc:{:.6f}'
			  .format(precision, recall, F1, accuracy))
		print("test:")
		predict_test = LRmodel.predict(X_test)
		accuracy, precision, recall, F1 = self.get_f1(predict_test, y_test)
		print('precision:{:.6f}\trecall:{:.6f}\tF1:{:.6f}\tacc:{:.6f}'
			  .format(precision, recall, F1, accuracy))
if __name__ == '__main__':
	args = read_args()
	logging.info("------arguments-------")
	for k, v in vars(args).items():
		logging.info(k + ': ' + str(v))

	#fix random seed
	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed)

	#model 
	model_object = model_class(args)
	#model_object.RF()
	#model_object.LR()

	if args.train_test_label == 0:
		#model_object.model_train()
		model_object.model_test()


