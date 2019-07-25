from __future__ import absolute_import

import statsmodels.api as sm

from .utils.parameters import *
from .data_processing import load_data
from .init_es import init_es
from .fuzzy_matching.candidate_graph import CandidateGraph
from .crf import tagger
from .feature_extraction import extract_features

import numpy as np
import pickle
import copy

model = None

def judge(raw_add, entities, candidates):
	global model
	X = []
	for candidate in candidates:
		X.append(extract_features(raw_add, entities, candidate))

	if model == None:
		model = pickle.load(open(MODEL_FINAL_FILE, 'rb'))
	y_preds = model.predict(X)
	print(y_preds)

	ret = []
	for i in range(len(candidates)):
		candidate = copy.deepcopy(candidates[i])
		candidate['score'] = y_preds[i][1]
		ret.append(candidate)
	return ret

def train():
	raw_data = load_data(TRAIN_FINAL_FILE)
	print('number of sample =', len(raw_data))

	X_train = []
	Y_train = []

	init_es()
	for raw_add, std_add in raw_data:
		graph = CandidateGraph.build_graph(raw_add)
		graph.prune_by_beam_search(k=BEAM_SIZE)
		candidates = graph.extract_address()
		crf_entities = tagger.detect_entity(raw_add)

		for candidate in candidates:
			X_train.append(extract_features(raw_add, crf_entities, candidate))
			Y_train.append(1 if int(std_add['id']) == int(candidate['addr_id']) else 0)

	print('length of X_train', len(X_train))

	model = sm.Poisson(Y_train, X_train)
	model = model.fit(method='newton')

	print('Model score: ', model.summary())

	preds = model.predict([X_train[0]])
	print(preds)
	acc = (Y_train == preds).mean()
	print('training accuracy = ', acc)

	pickle.dump(model, open(MODEL_FINAL_FILE, 'wb'))

if __name__ == '__main__':
	print(judge('doan ke thien cau giay ha noi'))