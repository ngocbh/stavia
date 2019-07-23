from __future__ import absolute_import

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from .utils.parameters import *
from .data_processing import load_data
from .init_es import init_es
from .fuzzy_matching.candidate_graph import CandidateGraph
from .crf import tagger
from .feature_extraction import extract_features

import numpy as np
import pickle


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

	model = LogisticRegression(C=1e20,verbose=1, fit_intercept=True, max_iter=1000)
	model.fit(X_train, Y_train)

	print('Model score',model.score(X_train, Y_train))

	preds = model.predict(X_train)
	acc = (Y_train == preds).mean()
	print('training accuracy = ', acc)

	print('Model parameters:')
	print(model.intercept_, model.coef_)

	pickle.dump(model, open(MODEL_FINAL_FILE, 'wb'))

if __name__ == '__main__':
	train()