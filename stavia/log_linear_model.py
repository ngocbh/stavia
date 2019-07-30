from __future__ import absolute_import

from .feature_set import FeatureSet
from scipy.optimize import fmin_l_bfgs_b
from math import exp, log
from sklearn.model_selection import train_test_split


from .utils.parameters import *
from .utils.utils import create_status
from .data_processing import load_data
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
	feature_set = None
	learning_rate = 0.01
	num_iter = 10000
	fit_intercept = True
	verbose = False

	# For L-BFGS
	GRADIENT = None
	squared_sigmas = [0.000001, 0.000003, 0.000009, \
					0.00001, 0.00003, 0.00009, \
					0.0001, 0.0003, 0.0009, \
					0.001, 0.003, 0.009, \
					0.01, 0.03, 0.09, \
					0.1, 0.3, 0.9,\
					1, 3, 9,\
					1e3, 3e3, 9e3,\
					1e5]

	def __init__(self, learning_rate=0.01, num_iter=10000, fit_intercept=True, verbose=False):
		self.learning_rate = learning_rate
		self.num_iter = num_iter
		self.fit_intercept = fit_intercept
		self.verbose = verbose
		self.feature_set = FeatureSet()

	def __softmax(self, potential):
		potential = np.exp(potential)
		Z = np.sum(potential)
		potential = potential / Z
		return potential, Z
	
	def __log_likelihood(self, params, *args):
		"""
		Calculate likelihood and gradient
		"""
		X, y, feature_set, squared_sigma, verbose, sign = args

		no_example = len(X)
		total_logZ = 0
		total_logProb = 0
		empirical_weights = np.zeros(len(feature_set))
		expected_weights = np.zeros(len(feature_set))

		for example_features, example_labels in zip(X, y):
			feature_table = np.asarray(feature_set.get_feature_table(example_features))
			potential = np.dot(feature_table, params)
			# print(np.sum(feature_table,axis=0))
			empirical_weights = empirical_weights + np.dot(feature_table.T , np.asarray(example_labels))
			#scaling 
			potential = potential - np.max(potential, keepdims=True)

			total_logProb += np.sum(np.multiply(potential, np.asarray(example_labels)))
			potential, Z = self.__softmax(potential)
			expected_weights = expected_weights + np.dot(potential.T, feature_table).T

			total_logZ += log(Z)

		log_likelihood = total_logProb - total_logZ - np.sum(np.multiply(params,params))/(squared_sigma*2)

		gradients = empirical_weights - expected_weights - params/squared_sigma
		self.GRADIENT = gradients

		global SUB_ITERATION_NUM
		if verbose:
			sub_iteration_str = '    '
			if SUB_ITERATION_NUM > 0:
				sub_iteration_str = '(' + '{0:02d}'.format(SUB_ITERATION_NUM) + ')'
			print('  ', '{0:03d}'.format(ITERATION_NUM), sub_iteration_str, ':', log_likelihood * sign)

		SUB_ITERATION_NUM += 1

		return sign * log_likelihood


	def __gradient(self, params, *args):
		"""
		Calculate gradient
		"""
		_, _, _, _, _,  sign = args
		return sign * self.GRADIENT



	def __estimate_parameters(self, X, y, squared_sigma, verbose):
		print('* Squared sigma:', squared_sigma)
		print('* Start L-BGFS')
		print('   ========================')
		print('   iter(sit): likelihood')
		print('   ------------------------')

		self.params, log_likelihood, information = \
				fmin_l_bfgs_b(func=self.__log_likelihood, fprime=self.__gradient,
							  x0=np.zeros(len(self.feature_set)),
							  args=(X, y, self.feature_set, squared_sigma, verbose, -1.0),
							  epsilon=1e-9,
							  maxls=self.num_iter,
							  callback=_callback)

		print('   ========================')
		print('   (iter: iteration, sit: sub iteration)')
		print('* Training has been finished with %d iterations' % information['nit'])

		if information['warnflag'] != 0:
			print('* Warning (code: %d)' % information['warnflag'])
			if 'task' in information.keys():
				print('* Reason: %s' % (information['task']))
		print('* Likelihood: %s' % str(log_likelihood))
	
	def fit(self, X, y, squared_sigma=0.001):
		start_time = time.time()
		print('[%s] Start training' % datetime.datetime.now())

		self.feature_set.scan(X)
		print('Number of feature = ', len(self.feature_set))

		self.__estimate_parameters(X, y, squared_sigma, verbose = self.verbose)

		print(self.params)
		elapsed_time = time.time() - start_time
		print('* Elapsed time: %f' % elapsed_time)
		print('* [%s] Training done' % datetime.datetime.now())

	def fit_regularized(self, X_train, y_train, X_dev, y_dev):
		start_time = time.time()
		print('[%s] Start training' % datetime.datetime.now())

		self.feature_set.scan(X_train)
		print('Number of feature = ', len(self.feature_set))

		max_acc = 0
		best_squared_sigma = -1
		for squared_sigma in self.squared_sigmas:
			self.__estimate_parameters(X_train, y_train, squared_sigma, verbose = False)
			acc = self.score(X_dev, y_dev)
			if acc > max_acc:
				max_acc = acc
				best_squared_sigma = squared_sigma
			print('testing', squared_sigma, acc)

		print('Choose Squared Sigma = ' , best_squared_sigma)
		print('Final Round')
		self.__estimate_parameters(X_train, y_train, best_squared_sigma, verbose = False)
		acc = self.score(X_train, y_train)
		print('Training Score = ', acc)
		acc = self.score(X_dev, y_dev)
		print('Development Score = ', acc)

	
	def predict_proba(self, example_features):
		feature_table = self.feature_set.get_feature_table(example_features)
		potential = np.dot(feature_table, self.params)
		prob, _ = self.__softmax(potential)
		return list(prob)

	def predict(self, example_features):
		prob = self.predict_proba(example_features)
		max_prob = max(prob)
		max_index = prob.index(max_prob)
		return max_prob, max_index

	def score(self, X_test, y_test):
		true_example = 0
		total_example = 0
		for example_features, example_labels in zip(X_test, y_test):
			prob, index = self.predict(example_features)
			if example_labels[index] == 1:
				true_example += 1
			total_example += 1
		return true_example / float(total_example)


def train():
	raw_data = load_data(TRAIN_FINAL_FILE)
	print('number of sample =', len(raw_data))
	sys.stdout.flush()

	X_data = []
	Y_data = []

	print('Extracing Feature -----> ')
	sys.stdout.flush()

	init_es()
	status = iter(create_status(len(raw_data)))

	number_positive_sample = 0
	for raw_add, std_add in raw_data:
		graph = CandidateGraph.build_graph(raw_add)
		graph.prune_by_beam_search(k=BEAM_SIZE)
		candidates = graph.extract_address()
		words, labels = tagger.tag(raw_add)
		crf_entities = tagger.detect_entity(raw_add, words, labels)
		example_features = []
		example_labels = []
		for candidate in candidates:
			example_features.append(extract_features(raw_add, words, labels, crf_entities, candidate))
			example_labels.append(1 if str(candidate['addr_id']) in std_add else 0)
			number_positive_sample += example_labels[-1]
		X_data.append(example_features)
		Y_data.append(example_labels)
		next(status)

	with codecs.open('x.json', encoding='utf8', mode='w') as f:
		js_str = json.dumps(X_data, ensure_ascii=False, indent=4)
		f.write(js_str)

	with codecs.open('y.json', encoding='utf8', mode='w') as f:
		js_str = json.dumps(Y_data, ensure_ascii=False, indent=4)
		f.write(js_str)

	print('Number Positive sample = ', number_positive_sample)
	print('Number Sample = ', len(Y_data))
	print('Spliting data')
	sys.stdout.flush()

	X_train, X_dev, y_train, y_dev = train_test_split(X_data, Y_data, test_size=0.13, random_state=42)

	model = LogLinearModel(learning_rate=0.0001, num_iter=3000, verbose=True)
	# model.fit_regularized(X_train, y_train, X_dev, y_dev)
	model.fit(X_train, y_train)
	print('Training score = ', model.score(X_train, y_train))
	print('Development score = ', model.score(X_dev, y_dev))

	print('Model parameters:')
	print(model.params)

	pickle.dump(model, open(MODEL_FINAL_FILE, 'wb'))
	
# if __name__ == '__main__':
	# print(judge('doan ke thien cau giay ha noi'))
		
