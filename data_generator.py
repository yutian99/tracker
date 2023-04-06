
import numpy as np
import re
import json
import random


class input_data(object):
	def __init__(self, args):
		self.args = args
		self.url2idx={}
		cnt=0
		if self.args.train_test_label != 2:  #train | test
			# store domain name pre-trained embedding
			d_name_embed = np.zeros((self.args.D_n, self.args.in_f_d))
			d_n_e_f = open(self.args.data_path + "bert_embedding_new_20000.txt", "r")
			lines=d_n_e_f.readlines()
			for line in lines:
				line=line.strip()
				url = re.split(' ', line)[0]
				length=len(url)
				line=line[length:].strip()[1:-1].strip()
				embeds=np.asarray(line.split(","),dtype="float32")
				self.url2idx[url]=cnt
				d_name_embed[cnt] = embeds
				cnt+=1
			d_n_e_f.close()

			self.d_name_embed = d_name_embed

			#store domain graph pre-trained embedding
			d_graph_embed = np.zeros((self.args.D_n, 128))
			d_g_e_f = open(self.args.data_path + "graph_embedding_new_20000_len20_128.txt", "r")
			lines=d_g_e_f.readlines()
			for line in lines:
				line=line.strip()
				url = re.split(' ', line)[0]
				if url not in self.url2idx:
					#print(url)
					continue
				embeds=np.asarray(line.split(" ")[1:],dtype="float32")
				d_graph_embed[self.url2idx[url]] = embeds
			d_g_e_f.close()

			self.d_graph_embed = d_graph_embed


            

			d_labels = np.zeros(self.args.D_n)
			self.tracker_idx = []
			self.normal_idx = []
			tracker=0
			d_labels_f = open(self.args.data_path + "id_url_label_new_20000.txt", "r")
			for line in d_labels_f:
				line = line.strip()
				url = re.split(' ', line)[0]
				label=int(re.split(' ', line)[2])
				d_labels[self.url2idx[url]]=label
				if label==1:
					tracker+=1
					self.tracker_idx.append(self.url2idx[url])
				else:
					self.normal_idx.append(self.url2idx[url])

			self.tracker=tracker
			self.d_labels=d_labels










