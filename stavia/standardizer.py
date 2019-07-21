from __future__ import absolute_import

from sklearn.externals import joblib
from .init_es import init_es
from .fuzzy_matching.candidate_graph import CandidateGraph
from .crf import tagger
from .feature_extraction import extract_features
from .utils.parameters import *

import copy

def rerank(raw_add, entities, candidates):
	X = []
	for candidate in candidates:
		X.append(extract_features(raw_add, entities, candidate))

	model = joblib.load(MODEL_FINAL_FILE)
	y_preds = model.predict_proba(X)

	ret = []
	for i in range(len(candidates)):
		candidate = copy.deepcopy(candidates[i])
		candidate['score'] = y_preds[i][1]
		ret.append(candidate)
	return ret


def standardize(addr):
	init_es()

	graph = CandidateGraph.build_graph(addr)
	graph.prune_by_beam_search(k=BEAM_SIZE)

	candidates = graph.extract_address()
	crf_entities = tagger.detect_entity(addr)

	ranked_list = rerank(addr, crf_entities, candidates)
	result = max(ranked_list, key=lambda element: element['score'])
	return result

