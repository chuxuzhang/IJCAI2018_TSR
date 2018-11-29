import tensorflow as tf
import keras
from keras.preprocessing import sequence
import cPickle as pkl
import data_generator
from itertools import *
import argparse
import os

# input arguments
parser = argparse.ArgumentParser(description='demo code of TSR')

parser.add_argument('--author_num', type = int, default = 28649,
				   help = 'max id of author')

parser.add_argument('--paper_num', type = int, default = 21046,
				   help = 'max id of paper')

parser.add_argument('--embed_dim', type = int, default = 128,
				   help = 'embed dimension of author and paper')

# parser.add_argument('--hidden_n', type = int, default = 128,
#                    help = 'hidden dimension of GRU encoder')

parser.add_argument('--model_path', type=str, default='../TSR/',
				   help='path to save model')

parser.add_argument('--window', type = int, default = 5,
				   help = 'window size for indirect relation')

parser.add_argument('--c_len', type = int, default = 100,
				   help = 'max len of paper content')

parser.add_argument('--batch_size', type = int, default = 500,
				   help = 'batch size of training')

parser.add_argument('--learn_rate', type = float, default = 0.001,
				   help = 'learning rate')

parser.add_argument('--p_neg_ids_num', type = float, default = 200,
				   help = 'number of candidate papers per author in evaluation')

parser.add_argument('--train_iter_max', type = float, default = 500,
				   help = 'max number of training iterations')

parser.add_argument('--save_model_freq', type = float, default = 10,
				   help = 'number of iterations to save model')

parser.add_argument('--c_reg', type = float, default = 0.001,
				   help = 'coefficient of regularization')

parser.add_argument('--margin_d', type = float, default = 0.1,
				   help = 'margin distance of augmented component')

parser.add_argument('--c_tradeoff', type = float, default = 0.1,
				   help = 'tradeoff coefficient of augmented component')

parser.add_argument('--data_path', type = str, default ='../data/',
				   help='path to data')

parser.add_argument('--train_test_label', type= int, default = 0,
				   help='train/test label: 0 - train, 1 - test, 2 - tf graph test')

parser.add_argument('--top_K', type= int, default = 10,
				   help='length of return list per author in evaluation')


args = parser.parse_args()
print(args)

# parameters setting
author_n = args.author_num
paper_n = args.paper_num
top_K = args.top_K

embed_d = args.embed_dim
hidden_n = args.embed_dim
c_len = args.c_len
c_reg = args.c_reg
margin_d = args.margin_d
c_tradeoff = args.c_tradeoff

batch_s = args.batch_size
lr = args.learn_rate
iter_max = args.train_iter_max
save_freq = args.save_model_freq

data_path = args.data_path
model_path = args.model_path

train_test_label = args.train_test_label

# data preparation
input_data = data_generator.input_data(args = args)
word_embed = input_data.word_embed

# generate negative paper ids in evaluation
if train_test_label == 2:
	input_data.gen_evaluate_neg_ids()

if train_test_label == 0 or train_test_label == 1:
	# tensor preparation
	# direct and indirect relation triples
	a_p_p_dir = tf.placeholder(tf.int32, [None, 3])
	a_p_p_indir = tf.placeholder(tf.int32, [None, 3])
	# paper content in direct and indirect relations
	p_c_dir_input = tf.placeholder(tf.int32, [None, c_len])
	p_c_indir_input = tf.placeholder(tf.int32, [None, c_len])

	# latent features/parameters of author
	author_embed = tf.Variable(tf.random_normal([author_n, embed_d], mean = 0, stddev = 0.01), name = "a_latent_pars")
	# pretrain word embedding of paper content
	p_c_dir_word_e = tf.cast(tf.nn.embedding_lookup(word_embed, p_c_dir_input), tf.float32)
	p_c_indir_word_e = tf.cast(tf.nn.embedding_lookup(word_embed, p_c_indir_input), tf.float32)
	# GRU encoder 
	with tf.variable_scope("text_rnn_encoder"):
		cell = tf.contrib.rnn.GRUCell(hidden_n)
		p_c_dir_deep_e, dir_state = tf.nn.dynamic_rnn(cell, p_c_dir_word_e, dtype = tf.float32)
	with tf.variable_scope("text_rnn_encoder", reuse=True):
		p_c_indir_deep_e, indir_state = tf.nn.dynamic_rnn(cell, p_c_indir_word_e, dtype = tf.float32)
	p_c_dir_e = tf.reduce_mean(p_c_dir_deep_e, 1)
	p_c_indir_e = tf.reduce_mean(p_c_indir_deep_e, 1)

	# accumuate loss
	# loss of direct relation
	Loss_1 = []
	for i in range(batch_s):
		a_e = tf.gather(author_embed, a_p_p_dir[i][0])
		a_e = tf.reshape(a_e, [1, embed_d])
		p_e_pos = tf.gather(p_c_dir_e, i*2)
		p_e_pos = tf.reshape(p_e_pos, [1, embed_d])
		p_e_neg = tf.gather(p_c_dir_e, i*2 + 1)
		p_e_neg = tf.reshape(p_e_neg, [1, embed_d])

		# pairwise ranking loss
		diff = tf.reduce_sum(tf.multiply(a_e, p_e_pos)) - tf.reduce_sum(tf.multiply(a_e, p_e_neg))
		Loss_1.append(-tf.log(tf.sigmoid(diff)))

	# loss of indirect relation
	# Loss_2 = []
	# for j in range(batch_s):
	# 	a_e = tf.gather(author_embed, a_p_p_indir[j][0])
	# 	a_e = tf.reshape(a_e, [1, embed_d])
	# 	p_e_pos = tf.gather(p_c_indir_e, j*2)
	# 	p_e_pos = tf.reshape(p_e_pos, [1, embed_d])
	# 	p_e_neg = tf.gather(p_c_indir_e, j*2 + 1)
	# 	p_e_neg = tf.reshape(p_e_neg, [1, embed_d])

	# 	# distance margin loss
	# 	hinge_l = tf.maximum(margin_d + tf.reduce_sum(tf.multiply(a_e, p_e_neg)) - tf.reduce_sum(tf.multiply(a_e, p_e_pos)), tf.zeros([1, 1]))
	# 	Loss_2.append(hinge_l)

	# loss formulation
	t_v = tf.trainable_variables()
	reg_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in t_v])
	# TSR
	joint_loss = tf.reduce_sum(Loss_1) + c_reg * reg_loss
	# TSR+
	#joint_loss = tf.reduce_sum(Loss_1) + c_tradeoff * tf.reduce_sum(Loss_2) + c_reg * reg_loss

	# optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = lr).minimize(joint_loss)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver(max_to_keep = 3)

# train/test 
if train_test_label == 0:# train model
	with tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads = 6,
			intra_op_parallelism_threads = 6)) as sess:
		sess.run(init)
		for epoch in range(iter_max):
			print("epoch: "+str(epoch))
			a_p_p_dir_batch = input_data.a_p_p_dir_next_batch()
			a_p_p_indir_batch = input_data.a_p_p_indir_next_batch()

			mini_batch_n = int(len(a_p_p_dir_batch)/batch_s)

			# divide each iteration into some mini batches 
			for i in range(mini_batch_n):
				a_p_p_dir_mini_batch = a_p_p_dir_batch[i*batch_s:(i+1)*batch_s]
				p_c_dir_mini_batch = input_data.gen_content_mini_batch(a_p_p_dir_mini_batch)
				a_p_p_indir_mini_batch = a_p_p_indir_batch[i*batch_s:(i+1)*batch_s]
				p_c_indir_mini_batch = input_data.gen_content_mini_batch(a_p_p_indir_mini_batch)

				feed_dict = {a_p_p_dir: a_p_p_dir_mini_batch, p_c_dir_input: p_c_dir_mini_batch, \
				a_p_p_indir: a_p_p_indir_mini_batch, p_c_indir_input: p_c_indir_mini_batch}
				_, loss_v = sess.run([optimizer, joint_loss], feed_dict)

				if i == 0:
					print("loss_value: "+str(loss_v))

			# last mini batcha_p_p_dir_mini_batch = a_p_p_dir_batch[len(a_p_p_dir_batch) - batch_s:len(a_p_p_dir_batch)]
			
			p_c_dir_mini_batch = input_data.gen_content_mini_batch(a_p_p_dir_mini_batch)
			a_p_p_indir_mini_batch = a_p_p_indir_batch[len(a_p_p_indir_batch) - batch_s:len(a_p_p_indir_batch)]
			p_c_indir_mini_batch = input_data.gen_content_mini_batch(a_p_p_indir_mini_batch)

			feed_dict = {a_p_p_dir: a_p_p_dir_mini_batch, p_c_dir_input: p_c_dir_mini_batch, \
			a_p_p_indir: a_p_p_indir_mini_batch, p_c_indir_input: p_c_indir_mini_batch}
			_, loss_v = sess.run([optimizer, joint_loss], feed_dict)

			# save model for evaluation
			if epoch % save_freq == 0:
				if not os.path.exists(model_path):
					os.makedirs(model_path)
				saver.save(sess, model_path + "TSR" + str(epoch) + ".ckpt")

				# evaluation tracking during training
				p_text_all = input_data.p_content
				p_text_deep_f = sess.run([p_c_dir_e], {p_c_dir_input: p_text_all})
				p_text_deep_f = p_text_deep_f[0]

				a_latent_f = tf.get_default_graph().get_tensor_by_name("a_latent_pars:0")
				a_latent_f = a_latent_f.eval()

				input_data.TSR_evaluate(p_text_deep_f, a_latent_f, top_K)

elif train_test_label == 1:# test model
	with tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads = 6,
			intra_op_parallelism_threads = 6)) as sess:
		restore_idx = 10
		saver.restore(sess, model_path + "TSR" + str(restore_idx) + ".ckpt")

		# load paper semantic deep embedding by learned rnn encoder
		p_text_all = input_data.p_content
		p_text_deep_f = sess.run([p_c_dir_e], {p_c_dir_input: p_text_all})
		p_text_deep_f = p_text_deep_f[0]

		# load learned author latent features/parameters
		a_latent_f = tf.get_default_graph().get_tensor_by_name("a_latent_pars:0")
		a_latent_f = a_latent_f.eval()

		# model evaluation 
		input_data.TSR_evaluate(p_text_deep_f, a_latent_f, top_K)
else:
	print "tf graph test finish."

		


