import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import read_args
import numpy as np
import string
import re
import math
args = read_args()


class HetAgg(nn.Module):
	def __init__(self, args, feature_list):
		super(HetAgg, self).__init__()
		self.embed_d = args.embed_d
		self.args = args 
		self.D_n = args.D_n
		self.feature_list = feature_list

		self.dd_neigh_att = nn.Parameter(torch.ones(512, 1), requires_grad = True)

		self.softmax = nn.Softmax(dim = 1)
		self.act = nn.LeakyReLU()


		self.conv_1 = nn.Conv1d(in_channels=1,out_channels=16,kernel_size=2)
		self.sig_1 = nn.Sigmoid()
		self.pool_1 = nn.MaxPool1d(2)
		self.bn_1 = nn.BatchNorm1d(16)
		self.conv_2 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=2)
		self.sig_2 = nn.Sigmoid()
		self.bn_2 = nn.BatchNorm1d(32)
		self.pool_2 = nn.MaxPool1d(4)
		self.conv_3 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=2)
		self.bn_3 = nn.BatchNorm1d(64)
		self.sig_3 = nn.Sigmoid()
		self.pool_3 = nn.MaxPool1d(8)
		self.conv_4 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=2)
		self.bn_4 = nn.BatchNorm1d(64)
		self.pool_4 = nn.MaxPool1d(8)
		self.sig_4 = nn.Sigmoid()
		self.flatten = nn.Flatten()
		self.fc_1 = nn.Linear(128,512)
		self.fc_dp = nn.Dropout(p = self.args.p1)
		self.fc_2 = nn.Linear(768,512)
		self.fc_dp2 = nn.Dropout(p=self.args.p2)
		self.fc_dropout = nn.Dropout(p =self.args.p3)
		self.fc_dropout2 = nn.Dropout(p = self.args.p4)
		self.fc_dropout3 = nn.Dropout(p = self.args.p5)
		#self.fc_linear = nn.Linear(16*383, 1024)
		self.fc_linear = nn.Linear(512, 1024)
		self.fc_linear2 = nn.Linear(1024, 2)
		#self.fc_linear3 = nn.Linear(2048,2)
		self.fc_soft=nn.Softmax(dim=1)


	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
				nn.init.xavier_normal_(m.weight.data)
				#nn.init.normal_(m.weight.data)
				m.bias.data.fill_(0.1)

	def node_het_agg(self, id_batch): #heterogeneous neighbor aggregation

		#attention module
		id_batch = np.reshape(id_batch, (1, -1))
		bert_embed_batch=self.feature_list[1][id_batch]
		net_embed_batch = self.feature_list[0][id_batch]
		#print(net_embed_batch.shape)
		#'''
		net_embed_batch = self.fc_dp(net_embed_batch)
		net_embed_batch = self.act(self.fc_1(net_embed_batch))
		bert_embed_batch = self.fc_dp2(bert_embed_batch)
		bert_embed_batch = self.act(self.fc_2(bert_embed_batch))

		#compute weights
		concate_embed = torch.cat((bert_embed_batch, net_embed_batch), 1).view(len(bert_embed_batch), 2, self.embed_d )
		#concate_embed = torch.cat((bert_embed_batch, net_embed_batch), 1)

		#print(concate_embed.shape)  #(32,4,128)

		atten_w = self.act(torch.bmm(concate_embed, self.dd_neigh_att.unsqueeze(0).expand(len(bert_embed_batch), \
																							 *self.dd_neigh_att.size())))
		atten_w = self.softmax(atten_w).view(len(bert_embed_batch), 1, 2)

		#weighted combination

		weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(bert_embed_batch), self.embed_d)

		return weight_agg_batch




	def forward(self, id_batch,embed_d):
		#print("forward:")
		#print(type(id_batch)) #list
		#print(len(id_batch)) #32
		out_embeds =self.act(self.node_het_agg(id_batch))
		batch_size = len(id_batch)
		# make out_embeds 3D tensor. Batch_size * 1 * embed_d
		#out_embeds = out_embeds.view(batch_size, 1, embed_d)
		#print(out_embeds.shape)
		#out_embeds = self.conv_1(out_embeds)
		#out_embeds = self.bn_1(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.sig_1(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.pool_1(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.conv_2(out_embeds)
		#out_embeds = self.bn_2(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.sig_2(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.pool_2(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.conv_3(out_embeds)
		#out_embeds = self.bn_3(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.sig_3(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.pool_3(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.conv_4(out_embeds)
		#out_embeds = self.bn_4(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.sig_4(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.pool_4(out_embeds)
		#print(out_embeds.shape)
		#out_embeds = self.flatten(out_embeds)
		#print(out_embeds.shape)
		out_embeds = self.fc_dropout(out_embeds)
		out_embeds = self.act(self.fc_linear(out_embeds))
		out_embeds = self.fc_dropout2(out_embeds)
		out_embeds = self.act(self.fc_linear2(out_embeds))
		#out_embeds = self.fc_dropout3(out_embeds)
		#out_embeds = self.act(self.fc_linear3(out_embeds))
		#predictions = self.fc_soft(out_embeds)  # log(1/(1+exp(-x)))    sigmoid = 1/(1+exp(-x))
		return out_embeds
		#print(predictions.shape)
		#return predictions


def cross_entropy_loss(out_embeds,id_batch,labels):
	citeration=nn.CrossEntropyLoss()
	fc_soft=nn.Softmax(dim=1)
	mini_labels=[]
	TP=0
	TN=0
	FN=0
	FP=0
	for i in id_batch:
		mini_labels.append(labels[i])
	#out_embeds_tmp=out_embeds.detach().numpy()
	#predictions=np.argmax(out_embeds_tmp, axis=1)
	predictions=fc_soft(out_embeds)
	_,predictions=torch.max(predictions,1)
	#print(predictions.shape)
	for i in range(len(predictions)):
		if predictions[i]==1 and mini_labels[i]==1:  #恶意域名预测正确
			TP+=1
		if predictions[i] == 0 and mini_labels[i] == 0: #benign域名预测正确
			TN+=1
		if predictions[i] == 0 and mini_labels[i] == 1: #恶意域名预测错误
			FN+=1
		if predictions[i] == 1 and mini_labels[i] == 0: #benign域名预测错误
			FP+=1
	target=torch.Tensor(mini_labels)
	target = target.type(torch.LongTensor)
	target = target.cuda()
	#print(predictions.is_cuda)
	#print(target.is_cuda)
	loss_sum = citeration(out_embeds, target)



	return loss_sum,TP,TN,FN,FP

