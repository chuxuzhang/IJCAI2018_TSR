import six.moves.cPickle as pickle
#import pandas as pd
import numpy as np
import string
import re
import random
from keras.preprocessing import sequence
from itertools import *

class input_data(object):
	def __init__(self, args):
		self.args = args

		# direct author-paper relation in train/test data
		a_p_dir_list_train = [[] for k in range(self.args.author_num)]
		a_p_dir_list_test = [[] for k in range(self.args.author_num)]
		p_train_label = [0] * self.args.paper_num
		dir_relation_f = ["author_paper_list_train.txt", "author_citation_list_train.txt",\
						  "author_paper_list_test.txt", "author_citation_list_test.txt"]
		for i in range(len(dir_relation_f)):
			f_name = dir_relation_f[i]
			neigh_f = open(self.args.data_path + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				node_id = string.atoi(re.split(':', line)[0])
				neigh_list = re.split(':', line)[1]
				neigh_list_id = re.split(',', neigh_list)
				if f_name == "author_paper_list_train.txt" or f_name == "author_citation_list_train.txt":
					for j in range(len(neigh_list_id)):
						a_p_dir_list_train[node_id].append('p'+str(neigh_list_id[j]))
						p_train_label[int(neigh_list_id[j])] = 1
				else:
					for k in range(len(neigh_list_id)):
						a_p_dir_list_test[node_id].append('p'+str(neigh_list_id[k]))
			neigh_f.close()

		self.a_p_dir_list_train = a_p_dir_list_train
		self.a_p_dir_list_test = a_p_dir_list_test
		self.dir_len = sum(len(x) for x in self.a_p_dir_list_train)
		self.p_train_label = p_train_label
		
		# indirect author-paper relation from heterogeneous walk
		a_p_indir_list_train = [[] for k in range(self.args.author_num)]
		def a_p_indir_set(path):
			het_walk_f = open(self.args.data_path + "het_random_walk.txt", "r")
			for line in het_walk_f:
				line=line.strip()
				path = re.split(' ',line)
				for k in range(len(path)):
					curr_node = path[k]
					if curr_node[0] == 'a':
						for w in range(k - self.args.window, k + self.args.window +1):
							if w >= 0 and w < len(path) and w != k:
								neigh_node = path[w]
								if neigh_node[0] == 'p' and neigh_node not in self.a_p_dir_list_train[int(curr_node[1:])]:
									a_p_indir_list_train[int(curr_node[1:])].append(neigh_node)
			het_walk_f.close()
			return a_p_indir_list_train

		self.a_p_indir_list_train = a_p_indir_set(self.args.data_path)
		self.indir_len = sum(len(x) for x in self.a_p_indir_list_train)

		def load_p_content(path, word_n = 100000):
			f = open(path, 'rb')
			p_content_set = pickle.load(f)
			f.close()

			def remove_unk(x):
				return [[1 if w >= word_n else w for w in sen] for sen in x]

			p_content, p_content_id = p_content_set
			p_content = remove_unk(p_content)
			p_content_set = (p_content, p_content_id)

			return p_content_set

		def load_word_embed(path, word_n = 54565, word_dim = 128):
			word_embed = np.zeros((word_n + 2, word_dim))

			f = open(path,'r')
			for line in islice(f, 1, None):
				index = int(line.split()[0])
				embed = np.array(line.split()[1:])
				word_embed[index] = embed

			return word_embed

		# text content (e.g., abstract) of paper and pretrain word embedding 
		self.p_content, self.p_content_id = load_p_content(path = self.args.data_path + 'content.pkl')
		self.p_content = sequence.pad_sequences(self.p_content, maxlen = self.args.c_len, value = 0., padding = 'post', truncating = 'post') 
		self.word_embed = load_word_embed(path = self.args.data_path + 'word_embedding.txt')

	def a_p_p_dir_next_batch(self):
		a_p_p_dir_list_batch = []
		for i in range(self.args.author_num):
			for j in range(len(self.a_p_dir_list_train[i])):
				p_neg = random.randint(0, self.args.paper_num - 1)
				while (('p'+str(p_neg)) in self.a_p_dir_list_train[i]):
					p_neg = random.randint(0, self.args.paper_num - 1)
				p_pos = int(self.a_p_dir_list_train[i][j][1:])
				triple=[i, p_pos, p_neg]
				a_p_p_dir_list_batch.append(triple)
		return a_p_p_dir_list_batch

	def a_p_p_indir_next_batch(self):
		a_p_p_indir_list_batch = []
		p_threshold = float(self.dir_len)/self.indir_len + 3e-3

		for i in range(self.args.author_num):
			for j in range(len(self.a_p_indir_list_train[i])):
				if random.random() < p_threshold:
					p_neg = random.randint(0, self.args.paper_num - 1)
					while (('p'+str(p_neg)) in self.a_p_dir_list_train[i]):
						p_neg = random.randint(0, self.args.paper_num - 1)
					p_pos = int(self.a_p_indir_list_train[i][j][1:])
					triple=[i, p_pos, p_neg]
					a_p_p_indir_list_batch.append(triple)
		return a_p_p_indir_list_batch

	def gen_content_mini_batch(self, triple_batch):
		p_c_data = []
		for i in range(len(triple_batch)):
			pos_c = (self.p_content[triple_batch[i][1]]).reshape(self.args.c_len)
			neg_c = (self.p_content[triple_batch[i][2]]).reshape(self.args.c_len)
			p_c_data.append(pos_c)
			p_c_data.append(neg_c)
		return p_c_data

	def gen_evaluate_neg_ids(self):
		paper_n_ave = 0
		author_n = 0
		a_p_neg_ids_f = open(self.args.data_path + "author_paper_neg_ids.txt", "w")
		for i in range(self.args.author_num):
			if len(self.a_p_dir_list_test[i]) > 2 and len(self.a_p_dir_list_train[i]):
				a_p_neg_ids_f.write(str(i) + ":")
				neg_num = self.args.p_neg_ids_num - len(self.a_p_dir_list_test[i])
				for j in range(neg_num):
					neg_id = random.randint(0, self.args.paper_num - 1)
					neg_id_str = 'p' + str(neg_id)
					while (neg_id_str in self.a_p_dir_list_test[i] or neg_id_str in self.a_p_dir_list_train[i]):
						neg_id = random.randint(0, self.args.paper_num - 1)
						neg_id_str = 'p' + str(neg_id)
					a_p_neg_ids_f.write(str(neg_id) + ",")
				a_p_neg_ids_f.write("\n")
				paper_n_ave += len(self.a_p_dir_list_test[i])
				author_n += 1
		a_p_neg_ids_f.close()
		print ("author_n_ave_test: " + str(float(paper_n_ave)/author_n))

	def TSR_evaluate(self, p_text_deep_f, a_latent_f, top_K):
		a_p_neg_list_test = [[] for k in range(self.args.author_num)]
		a_p_neg_ids_f = open(self.args.data_path + "author_paper_neg_ids.txt", "r")
		for line in a_p_neg_ids_f:
			line = line.strip()
			a_id = string.atoi(re.split(':', line)[0])
			p_list = re.split(':', line)[1]
			p_list_ids = re.split(',', p_list)
			for i in range(len(p_list_ids) - 1):
				a_p_neg_list_test[int(a_id)].append(int(p_list_ids[i]))
		a_p_neg_ids_f.close()

		# Recall/Precision Scores
		evaluate_a_num = 0
		recall_ave = 0
		pre_ave = 0

		for i in range(self.args.author_num):
			if len(self.a_p_dir_list_test[i]) > 2 and len(self.a_p_dir_list_train[i]) and len(a_p_neg_list_test[i]):
				evaluate_a_num += 1
				correct_num = 0
				score_list = []
				for j in range(len(self.a_p_dir_list_test[i])):
					p_id_temp = self.a_p_dir_list_test[i][j][1:]
					score_temp = np.dot(a_latent_f[i], p_text_deep_f[int(p_id_temp)])
					score_list.append(score_temp)
				for k in range(len(a_p_neg_list_test[i])):
					p_id_temp = a_p_neg_list_test[i][k]
					score_temp = np.dot(a_latent_f[i], p_text_deep_f[int(p_id_temp)])
					score_list.append(score_temp)
				score_list.sort()

				score_threshold = score_list[-top_K - 1]

				for jj in range(len(self.a_p_dir_list_test[i])):
					p_id_temp = self.a_p_dir_list_test[i][jj][1:]
					score_temp = np.dot(a_latent_f[i], p_text_deep_f[int(p_id_temp)])
					if score_temp > score_threshold:
						correct_num += 1

				recall_ave += float(correct_num) / len(self.a_p_dir_list_test[i])
				pre_ave += float(correct_num) / top_K

		print ("total evaluate author number: " + str(evaluate_a_num))
		recall_ave = recall_ave / evaluate_a_num
		pre_ave = pre_ave / evaluate_a_num
		print ("recall_ave@top_K: " + str(recall_ave))
		print ("pre_ave@top_K: " + str(pre_ave))

		# AUC Score
		AUC_ave = 0
		for i in range(self.args.author_num):
			if len(self.a_p_dir_list_test[i]) > 2 and len(self.a_p_dir_list_train[i]) and len(a_p_neg_list_test[i]):
				neg_score_list = []
				correct_num = 0
				pair_num = 0
				for k in range(len(a_p_neg_list_test[i])):
					p_id_temp = a_p_neg_list_test[i][k]
					score_temp = np.dot(a_latent_f[i], p_text_deep_f[int(p_id_temp)])
					neg_score_list.append(score_temp)

				for j in range(len(self.a_p_dir_list_test[i])):
					p_id_temp = self.a_p_dir_list_test[i][j][1:]
					pos_score = np.dot(a_latent_f[i], p_text_deep_f[int(p_id_temp)])
					for jj in range(len(neg_score_list)):
						pair_num += 1
						if pos_score > neg_score_list[jj]:
							correct_num += 1
				AUC_ave += float(correct_num) / pair_num

		AUC_ave = AUC_ave / evaluate_a_num
		print ("AUC_ave: " + str(AUC_ave))

