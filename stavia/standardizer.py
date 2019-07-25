from __future__ import absolute_import

from .init_es import init_es, reinit_es
from .fuzzy_matching.candidate_graph import CandidateGraph
from .crf import tagger
from .feature_extraction import extract_features
from .utils.parameters import *
from .logistic_regression import judge
import copy
import pickle

model = None

def standardize(addr):
	init_es()

	graph = CandidateGraph.build_graph(addr)
	graph.prune_by_beam_search(k=BEAM_SIZE)
	
	candidates = graph.extract_address()
	crf_entities = tagger.detect_entity(addr)
	ranked_list = judge(addr, crf_entities, candidates)
	# print(ranked_list)
	result = max(ranked_list, key=lambda element: element['score'])
	return result

