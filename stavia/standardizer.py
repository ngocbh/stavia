from __future__ import absolute_import

from .init_es import init_es, reinit_es
from .fuzzy_matching.candidate_graph import CandidateGraph
from .crf import tagger
from .feature_extraction import extract_features
from .utils.parameters import *
from .log_linear_model import LogLinearModel
from .logistic_regression import lr_judge
from .fuzzy_matching import sagel
import copy
import pickle

model = None

def llm_judge(addr, crf_entities, candidates):
	global model
	if model == None:
		model = pickle.load(open(MODEL_FINAL_FILE, 'rb'))

	example_features = []
	for candidate in candidates:
		example_features.append(extract_features(addr, crf_entities, candidate))

	y_prob = model.predict_proba(example_features)

	ret = []
	for i in range(len(candidates)):
		candidate = copy.deepcopy(candidates[i])
		candidate['score'] = y_prob[i]
		ret.append(candidate)
	return ret


def standardize(addr, method=METHOD):
	init_es()

	graph = CandidateGraph.build_graph(addr)
	graph.prune_by_beam_search(k=BEAM_SIZE)
	
	candidates = graph.extract_address()
	crf_entities = tagger.detect_entity(addr)
	print(crf_entities)
	if METHOD == 'lr':
		ranked_list = lr_judge(addr, crf_entities, candidates)
	else:
		ranked_list = llm_judge(addr, crf_entities, candidates)
	# print(ranked_list)
	if len(ranked_list) == 0:
		result = {'error': 'no candidate matched'}
	else: 
		result = max(ranked_list, key=lambda element: element['score'])
		
	return result


def llm_judge_4_testing(addr, crf_entities, candidates):
	global model
	if model == None:
		model = pickle.load(open(MODEL_FINAL_FILE, 'rb'))

	example_features = []
	for i in range(len(candidates)):
		candidate = candidates[i]
		candidate_features = extract_features(addr, crf_entities, candidate)
		example_features.append(candidate_features)
		candidate['features'] = {}
		for feature, value in candidate_features.items():
			candidate['features'][feature] = value

	y_prob = model.predict_proba(example_features)

	ret = []
	for i in range(len(candidates)):
		candidate = copy.deepcopy(candidates[i])
		candidate['score'] = y_prob[i]
		ret.append(candidate)
	return ret


def standardize4testing(addr, method=METHOD):
	init_es()

	graph = CandidateGraph.build_graph(addr)
	graph.prune_by_beam_search(k=BEAM_SIZE)
	
	candidates = graph.extract_address()
	words, labels = tagger.tag(addr)
	crf_entities = tagger.detect_entity(addr, words, labels)
	if METHOD == 'lr':
		ranked_list = lr_judge(addr, crf_entities, candidates)
	else:
		ranked_list = llm_judge_4_testing(addr, crf_entities, candidates)
		
	if len(ranked_list) == 0:
		result = {'error': 'no candidate matched'}
	else: 
		result = max(ranked_list, key=lambda element: element['score'])
	
	return result, ranked_list, crf_entities, words, labels


