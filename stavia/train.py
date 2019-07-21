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


def train():
	raw_data = load_data(TRAIN_FINAL_FILE)

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

	model = LogisticRegression(C=1e20,verbose=1,max_iter=200,solver='lbfgs')
	model.fit(X_train, Y_train)

	print('Training accuracy =',model.score(X_train, Y_train))

	joblib.dump(model, MODEL_FINAL_FILE)

if __name__ == '__main__':
	train()