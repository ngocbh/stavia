from __future__ import absolute_import

import codecs, json
import numpy as np

class FeatureSet:
	def __init__(self):
		self.feature_dict = dict()
		self.num_features = 0

	def __len__(self):
		return self.num_features

	def scan(self, X):
		training_features = []
		for example in X:
			hashed_example = []
			for candidate in example:
				hashed_candidate = {}
				for key, value in candidate.items():
					if key not in self.feature_dict:
						feature_id = self.num_features
						self.feature_dict.update({key: feature_id})
						self.num_features += 1
						

	def get_feature_vector(self, candidate_features):
		feature_vector = [0 for _ in range(self.num_features)]
		for feature_name, value in candidate_features.items():
			try:
				feature_id = self.feature_dict[feature_name]
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