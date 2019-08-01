from __future__ import absolute_import

import codecs, json
import numpy as np

class FeatureSet:
	def __init__(self):
		self.feature_dict = dict()
		self.num_features = 0
		self.empirical_weights = dict()

	def __len__(self):
		return self.num_features

	def scan(self, X, y):
		'''
		hash feature name and calculate empirical weights
		'''
		training_features = []
		for example, labels in zip(X,y):
			hashed_example = []
			for candidate, label in zip(example, labels):
				hashed_candidate = {}
				for key, value in candidate.items():
					if key not in self.feature_dict:
						feature_id = self.num_features
						self.feature_dict.update({key: feature_id})
						self.num_features += 1
					feature_id = self.feature_dict[key]
					hashed_candidate[feature_id] = value

					if feature_id not in self.empirical_weights:
						self.empirical_weights[feature_id] = value*label
					else:
						self.empirical_weights[feature_id] += value*label
				hashed_example.append(hashed_candidate)
			training_features.append(hashed_example)
		return training_features

	def hash_feature(self, example):
		hashed_example = []
		for candidate in example:
			hashed_candidate = {}
			for key, value in candidate.items():
				try:
					feature_id = self.feature_dict[key]
					hashed_candidate[feature_id] = value
				except KeyError:
					pass
			hashed_example.append(hashed_candidate)
		return hashed_example

	def get_empirical_weights(self):
		ret = np.zeros(self.num_features)
		for key, value in self.empirical_weights.items():
			ret[key] = value
		return ret
						
	def get_feature_vector(self, candidate_features):
		feature_vector = np.zeros(self.num_features)
		for feature_id, value in candidate_features.items():
			try:
				feature_vector[feature_id] = value
			except KeyError:
				pass
		return feature_vector

	def get_feature_table(self, example_features):
		feature_table = []
		for candidate_features in example_features:
			feature_table.append(self.get_feature_vector(candidate_features))

		return feature_table

	def calc_inner_product(self, candidate_features, params):
		"""
		Calculates inner products of the given parameters and feature vectors of the given observations.
		:param params: parameter vector
		:param X: observation vector
		:return: potential table 
		"""
		ret = 0
		for feature_id, value in candidate_features.items():
			try:
				ret += value * params[feature_id]
			except KeyError:
				pass
				
		return ret

	def calc_inner_sum(self, expected_weights, candidate_features, factor):
		"""
		Calculates inner sum ò the given expected_weights , candidate_features and factor is probability of candidate
		"""
		for feature_id, value in candidate_features.items():
			try:
				expected_weights[feature_id] += value * factor
			except KeyError:
				pass




