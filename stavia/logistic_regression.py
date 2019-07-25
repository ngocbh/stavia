from __future__ import absolute_import

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from .utils.parameters import *
from .data_processing import load_data
from .init_es import init_es
from .fuzzy_matching.candidate_graph import CandidateGraph
from .crf import tagger
from .feature_extraction import extract_features

import numpy as np
import pickle, copy

model = None

def judge(raw_add, entities, candidates):
	global model
	X = []
	for candidate in candidates:
		X.append(extract_features(raw_add, entities, candidate))

	if model == None:
		model = pickle.load(open(MODEL_FINAL_FILE, 'rb'))
	y_preds = model.predict_proba(X)

	ret = []
	for i in range(len(candidates)):
		candidate = copy.deepcopy(candidates[i])
		candidate['score'] = y_preds[i][1]
		ret.append(candidate)
	return ret

def train():
	raw_data = load_data(TRAIN_FINAL_FILE)
	print('number of sample =', len(raw_data))

	X_data = []
	Y_data = []

	print('Extracing Feature -----> ')
	init_es()
	for raw_add, std_add in raw_data:
		graph = CandidateGraph.build_graph(raw_add)
		graph.prune_by_beam_search(k=BEAM_SIZE)
		candidates = graph.extract_address()
		crf_entities = tagger.detect_entity(raw_add)

		for candidate in candidates:
			X_data.append(extract_features(raw_add, crf_entities, candidate))
			Y_data.append(1 if int(std_add['id']) == int(candidate['addr_id']) else 0)

	print('Spliting data')
	X_train, X_dev, Y_train, Y_dev = train_test_split(X_data, Y_data, test_size=0.13, random_state=42)
	print('length of X_train', len(X_train))

	lambs = [0.000001, 0.00001, 0.0001, 0.0003, 0.0006, 0.0001, 0.001, 0.003, 0.006, 0.01, 0.03, 1, 1e20]
	max_acc = 0
	best_lamb = 0.00001

	for lamb in lambs:
		print('Hyperparameters Tuning ------------------>>>')
		print('Lambda = ', lamb)
		model = LogisticRegression(C=lamb,verbose=1, fit_intercept=True, max_iter=1000)
		model.fit(X_train, Y_train)

		print('training score',model.score(X_train, Y_train))

		preds = model.predict(X_dev)
		acc = (Y_dev == preds).mean()
		print('validation accuracy = ', acc)

		if acc > max_acc:
			max_acc = acc
			best_lamb = lamb

	print("++++++++++++++ FINAL ROUND ++++++++++++")
	print("Choose lambda = ", best_lamb)
	model = LogisticRegression(C=best_lamb,verbose=0, fit_intercept=True, max_iter=1000)
	model.fit(X_train, Y_train)


	print('Model parameters:')
	print(model.intercept_, model.coef_)

	pickle.dump(model, open(MODEL_FINAL_FILE, 'wb'))

if __name__ == '__main__':
	print(judge('doan ke thien cau giay ha noi'))