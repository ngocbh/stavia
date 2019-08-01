from __future__ import absolute_import

from .feature_set import FeatureSet
from scipy.optimize import fmin_l_bfgs_b
from math import exp, log
from sklearn.model_selection import train_test_split


from .utils.parameters import *
from .utils.utils import create_status
from .data_processing import load_data, preprocess
from .init_es import init_es
from .fuzzy_matching.candidate_graph import CandidateGraph
from .crf import tagger
from .feature_extraction import extract_features

import json, codecs
import numpy as np
import time, datetime
import pickle, copy
import sys, re

ITERATION_NUM = 0
SUB_ITERATION_NUM = 0
TOTAL_SUB_ITERATIONS = 0

def _callback(params):
	global ITERATION_NUM
	global SUB_ITERATION_NUM
	global TOTAL_SUB_ITERATIONS
	ITERATION_NUM += 1
	TOTAL_SUB_ITERATIONS += SUB_ITERATION_NUM
	SUB_ITERATION_NUM = 0



class LogLinearModel:
	params = None
	lambda_reg = None
	feature_set = None
	learning_rate = 0.01
	num_iter = 10000
	fit_intercept = True
	verbose = False

	# For L-BFGS
	GRADIENT = None
	lambda_regs = [0.000001, 0.000003, 0.000009, \
					0.00001, 0.00003, 0.00009, \
					0.0001, 0.0003, 0.0009, \
					0.001, 0.003, 0.009, \
					0.01, 0.03, 0.09, \
					0.1, 0.3, 0.9,\
					1]

	def __init__(self, learning_rate=0.01, num_iter=10000, fit_intercept=True, verbose=False, lambda_reg=0.0001):
		self.learning_rate = learning_rate
		self.num_iter = num_iter
		self.fit_intercept = fit_intercept
		self.verbose = verbose
		self.feature_set = FeatureSet()
		self.lambda_reg = lambda_reg

	def __softmax(self, potential):
		potential = np.exp(potential)
		Z = np.sum(potential)
		potential = potential / Z
		return potential, Z
	
	def __log_likelihood(self, params, *args):
		"""
		Calculate likelihood and gradient
		"""
		X, y, feature_set, lambda_reg, empirical_weights, verbose, sign = args

		no_example = len(X)
		total_logZ = 0
		total_logProb = 0
		expected_weights = np.zeros(len(feature_set))
		for t in range(len(X)):
			# example_features = X[t], example_labels = y[t]

			potential = np.zeros(len(X[t]))
			for i in range(len(X[t])):
				#candidate_features = X[t][i], candidate_label = y[t][i]
				potential[i] = feature_set.calc_inner_product(X[t][i], params)

			#scaling
			potential = potential - np.max(potential, keepdims=True)

			for i in range(len(X[t])):
				total_logProb += potential[i] * y[t][i]

			potential, Z = self.__softmax(potential)

			for i in range(len(X[t])):
				feature_set.calc_inner_sum(expected_weights, X[t][i], potential[i])

			total_logZ += log(Z)


		log_likelihood = total_logProb - total_logZ - (lambda_reg/2) * np.sum(np.multiply(params,params))
		gradients = empirical_weights - expected_weights - lambda_reg * params

		global SUB_ITERATION_NUM
		if verbose:
			sub_iteration_str = '    '
			if SUB_ITERATION_NUM > 0:
				sub_iteration_str = '(' + '{0:02d}'.format(SUB_ITERATION_NUM) + ')'
			print('  ', '{0:03d}'.format(ITERATION_NUM), sub_iteration_str, ':', log_likelihood * sign)

		SUB_ITERATION_NUM += 1

		return sign * log_likelihood, sign * gradients


	def __gradient(self, params, *args):
		"""
		Calculate gradient
		"""
		_, _, _, _, _, _, sign = args
		return sign * self.GRADIENT


	def __estimate_parameters(self, X, y, lambda_reg, verbose):
		print('* Lambda:', lambda_reg)
		print('* Start L-BGFS')
		print('   ========================')
		print('   iter(sit): likelihood')
		print('   ------------------------')

		params, log_likelihood, information = \
				fmin_l_bfgs_b(func=self.__log_likelihood,
							  x0=np.zeros(len(self.feature_set)),
							  args=(X, y, self.feature_set, lambda_reg, 
							  	self.feature_set.get_empirical_weights(), verbose, -1.0),
							  maxls=100,
							  maxiter=self.num_iter,
							  callback=_callback)

		print('   ========================')
		print('   (iter: iteration, sit: sub iteration)')
		print('* Training has been finished with %d iterations' % information['nit'])

		if information['warnflag'] != 0:
			print('* Warning (code: %d)' % information['warnflag'])
			if 'task' in information.keys():
				print('* Reason: %s' % (information['task']))
		print('* Likelihood: %s' % str(log_likelihood))

		return params
	
	def fit(self, X, y):
		start_time = time.time()
		print('[%s] Start training' % datetime.datetime.now())

		X = self.feature_set.scan(X, y)
		print('Number of feature = ', len(self.feature_set))

		self.params = self.__estimate_parameters(X, y, self.lambda_reg, verbose = self.verbose)

		elapsed_time = time.time() - start_time
		print('* Elapsed time: %f' % elapsed_time)
		print('* [%s] Training done' % datetime.datetime.now())

	def fit_regularized(self, X_train, y_train, X_dev, y_dev):
		start_time = time.time()
		print('[%s] Start training' % datetime.datetime.now())
		X_train = self.feature_set.scan(X_train, y_train)

		max_acc = 0
		self.lambda_reg = -1
		for lambda_reg in self.lambda_regs:
			self.params = self.__estimate_parameters(X_train, y_train, lambda_reg, verbose = False)
			acc = self.score(X_dev, y_dev)
			if acc > max_acc:
				max_acc = acc
				self.lambda_reg = lambda_reg
			print('testing on = ', lambda_reg, acc)

		print('Choose hyperparameter for regularization, lambda = ' , self.lambda_reg)
		print('---------Final Round--------')
		self.params = self.__estimate_parameters(X_train, y_train, self.lambda_reg, verbose = False)
		acc = self.score(X_train, y_train, hashed=True)
		print('Training Score = ', acc)
		acc = self.score(X_dev, y_dev)
		print('Development Score = ', acc)

	
	def predict_proba(self, example_features, hashed=False):
		if hashed == False:
			example_features = self.feature_set.hash_feature(example_features)

		potential = np.zeros(len(example_features))
		for i in range(len(example_features)):
			potential[i] = self.feature_set.calc_inner_product(example_features[i], self.params)

		prob, _ = self.__softmax(potential)
		# print(prob)
		return list(prob)

	def predict(self, example_features, hashed=False):
		prob = self.predict_proba(example_features,hashed)
		max_prob = max(prob)
		max_index = prob.index(max_prob)
		return max_prob, max_index

	def score(self, X_test, y_test, hashed=False):
		true_example = 0
		total_example = 0
		for example_features, example_labels in zip(X_test, y_test):
			prob, index = self.predict(example_features,hashed)
			if example_labels[index] == 1:
				true_example += 1
			total_example += 1
		return true_example / float(total_example)

def train():
	raw_data = load_data(TRAIN_FINAL_FILE)
	print('number of sample =', len(raw_data))
	sys.stdout.flush()

	data = preprocess(raw_data)

	with codecs.open('new_training_data.json', encoding='utf8', mode='w') as f:
		jstr = json.dumps(data,ensure_ascii=False, indent=4)
		f.write(jstr)

	print('Extracing Feature -----> ')
	sys.stdout.flush()

	init_es()
	status = iter(create_status(len(data)))
	X_data = []
	Y_data = []

	number_positive_sample = 0
	for raw_add, std_add in data:
		if len(std_add) > 1:
			raise Exception('Too many positive candidate per example', raw_add, std_add)

		graph = CandidateGraph.build_graph(raw_add)
		graph.prune_by_beam_search(k=BEAM_SIZE)
		candidates = graph.extract_address()
		words, labels = tagger.tag(raw_add)
		crf_entities = tagger.detect_entity(raw_add, words, labels)
		example_features = []
		example_labels = []
		pos_per_ex = 0
		for candidate in candidates:
			example_features.append(extract_features(raw_add, words, labels, crf_entities, candidate))
			example_labels.append(1 if str(candidate['addr_id']) in std_add else 0)
			number_positive_sample += example_labels[-1]
			pos_per_ex += example_labels[-1]

		# if pos_per_ex == 0:
		# 	raise Exception('positive candidate per example != 1 ', raw_add, std_add, pos_per_ex)
		if pos_per_ex == 1:
			X_data.append(example_features)
			Y_data.append(example_labels)
		next(status)

	with codecs.open('x.json', encoding='utf8', mode='w') as f:
		jstr = json.dumps(X_data,ensure_ascii=False, indent=4)
		f.write(jstr)

	with codecs.open('y.json', encoding='utf8', mode='w') as f:
		jstr = json.dumps(Y_data,ensure_ascii=False, indent=4)
		f.write(jstr)

	print('Number Positive sample = ', number_positive_sample)
	print('Number Sample = ', len(Y_data))
	print('Spliting data')
	sys.stdout.flush()

	X_train, X_dev, y_train, y_dev = train_test_split(X_data, Y_data, test_size=0.13, random_state=42)

	model = LogLinearModel(lambda_reg=0.00001,num_iter=3000, verbose=True)
	# model.fit_regularized(X_train, y_train, X_dev, y_dev)
	model.fit(X_train, y_train)
	print('Training score = ', model.score(X_train, y_train))
	print('Development score = ', model.score(X_dev, y_dev))

	print('Model parameters:')
	print(model.params)

	pickle.dump(model, open(MODEL_FINAL_FILE, 'wb'))
	
# if __name__ == '__main__':
	# print(judge('doan ke thien cau giay ha noi'))
		
