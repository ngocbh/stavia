from __future__ import absolute_import

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from .utils.parameters import *
from .data_processing import load_data
from .init_es import init_es
from .fuzzy_matching.candidate_graph import CandidateGraph
from .crf import tagger
from .utils.utils import contains_Vietchar, no_accent_vietnamese
from nltk import ngrams

import numpy as np
import pickle, copy
import sys
import re
import numpy as np

model = None


def levenshtein_ratio_and_distance(s, t, ratio_calc = True):
	""" levenshtein_ratio_and_distance:
		Calculates levenshtein distance between two strings.
		If ratio_calc = True, the function computes the
		levenshtein distance ratio of similarity between two strings
		For all i and j, distance[i,j] will contain the Levenshtein
		distance between the first i characters of s and the
		first j characters of t
	"""
	rows = len(s)+1
	cols = len(t)+1
	distance = np.zeros((rows,cols),dtype = int)

	for i in range(1, rows):
		for k in range(1,cols):
			distance[i][0] = i
			distance[0][k] = k
  
	for col in range(1, cols):
		for row in range(1, rows):
			if s[row-1] == t[col-1]:
				cost = 0 
			else:
				if ratio_calc == True:
					cost = 2
				else:
					cost = 1
			distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
								 distance[row][col-1] + 1,          # Cost of insertions
								 distance[row-1][col-1] + cost)     # Cost of substitutions
	if ratio_calc == True:
		Ratio = ((len(s)+len(t)) - distance[row][col]) / float(len(s)+len(t))
		return Ratio
	else:
		return distance[row][col]

def get_ngrams(text, n):
	n_grams = ngrams(text, n)
	return [''.join(grams) for grams in n_grams]



def jaccard_similarity(string1, string2):
	sum = 0
	n_gram = 3
	list1 = get_ngrams(re.sub(r'[^\w\s]', '', string1.lower()).strip(), n_gram);
	list2 = get_ngrams(re.sub(r'[^\w\s]', '', string2.lower()).strip(), n_gram);
	intersection = len(list(set(list1).intersection(list2)))
	union = (len(list1) + len(list2)) - intersection
	if union == 0:
		return float(0)
	sum += float(intersection / union)
	return float(sum)

def c_score(string1, string2):
	list2 = string2.split(", ")
	c = 0
	for i in list2:
		if i in string1:
			c += len(i.split(" "))
	return 0

def extract_features(raw_add, entities, candidate):
	features = []
	#Bias
	features.append(1)

	#Admin_level in crf
	for field in FIELDS:
		if field in entities:
			features.append(1)
		else:
			features.append(0)

	#Admin_level in candidate
	for field in FIELDS:
		if field in candidate.keys():
			features.append(1)
		else:
			features.append(0)

	#Is contain vietnamese character
	features.append(1 if contains_Vietchar(raw_add) == True else 0)

	#Elastic Score
	for field in FIELDS:
		if field + '_score' in candidate.keys():
			features.append(float(candidate[field+'_score']))
		else:
			features.append(0.0)

	#Entity Score
	for field in FIELDS:
		if field not in entities or field not in candidate.keys():
			features.append(0.0)
		else:
			value = 0
			for entity in entities[field]:
				value = max(value, 1 if entity.lower() == candidate[field].lower() else 0)
			features.append(value)

	#Entity Score with no_accent_vietnamese
	for field in FIELDS:
		if field not in entities or field not in candidate.keys():
			features.append(0.0)
		else:
			value = 0
			for entity in entities[field]:
				value = max(value, 1 if no_accent_vietnamese(entity.lower()) == no_accent_vietnamese(candidate[field].lower()) else 0)
			features.append(value)

	#Jaccard Score
	for field in FIELDS:
		if field not in entities or field not in candidate.keys():
			features.append(0.0)
		else:
			value = 0
			for entity in entities[field]:
				value = max(value,jaccard_similarity(entity, candidate[field]))
			features.append(value)

	#Jaccard Score with no_accent_vietnamese
	for field in FIELDS:
		if field not in entities or field not in candidate.keys():
			features.append(0.0)
		else:
			value = 0
			for entity in entities[field]:
				value = max(value, jaccard_similarity(no_accent_vietnamese(entity.lower()), no_accent_vietnamese(candidate[field].lower())))
			features.append(value)

	#Levenshtein Score
	for field in FIELDS:
		if field not in entities or field not in candidate.keys():
			features.append(0.0)
		else:
			value = 0
			for entity in entities[field]:
				value = max(value, levenshtein_ratio_and_distance(entity.lower(), candidate[field].lower()))
			features.append(value)

	#Levenshtein Score with no_accent_vietnamese
	for field in FIELDS:
		if field not in entities or field not in candidate.keys():
			features.append(0.0)
		else:
			value = 0
			for entity in entities[field]:
				value = max(value, levenshtein_ratio_and_distance(no_accent_vietnamese(entity.lower()), no_accent_vietnamese(candidate[field].lower())))
			features.append(value)

	return features


def lr_detect_entity(inp, tokens=None, labels=None):
	if tokens == None or labels == None:
		tokens, labels = tagger.tag(inp)
	entities = {}

	n = len(tokens)
	buff = ''
	lbuff = ''
	isEntity = False

	for i in range(n):
		if (labels[i][0] == 'I'):
			buff += ' ' + tokens[i]
		else:
			if isEntity == True:
				key = lbuff.lower() 
				if key not in entities:
					entities[key] = [buff]
				else:
					entities[key].append(buff)

			buff = tokens[i]
			if labels[i][0] == 'B':
				if labels[i] == 'B_DIST':
					lbuff = 'DISTRICT'
				elif labels[i] == 'B_PRO':
					lbuff = 'NAME'
				else:
					lbuff = labels[i][2:]
				isEntity = True
			else:
				lbuff = labels[i]
				isEntity = False

	if isEntity == True:
		key = lbuff.lower() 
		if key not in entities :
			entities[key] = [buff]
		else:
			entities[key].append(buff)


	return entities

def lr_judge(raw_add, entities, candidates):
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
	sys.stdout.flush()

	X_data = []
	Y_data = []

	print('Extracing Feature -----> ')
	sys.stdout.flush()
	init_es()
	number_positive_sample = 0
	for raw_add, std_add in raw_data:
		graph = CandidateGraph.build_graph(raw_add)
		graph.prune_by_beam_search(k=BEAM_SIZE)
		candidates = graph.extract_address()
		crf_entities = lr_detect_entity(raw_add)

		for candidate in candidates:
			X_data.append(extract_features(raw_add, crf_entities, candidate))
			Y_data.append(1 if str(candidate['addr_id']) in std_add else 0)
			number_positive_sample += Y_data[-1]

	print('Number Positive sample = ', number_positive_sample)
	print('Number Sample = ', len(Y_data))
	print('Spliting data')
	sys.stdout.flush()

	X_train, X_dev, Y_train, Y_dev = train_test_split(X_data, Y_data, test_size=0.13, random_state=42)
	print('length of X_train', len(X_train))
	lambs = [0.000001, 0.00001, 0.0001, 0.0003, 0.0006, 0.0001, 0.001, 0.003, 0.006, 0.01, 0.03, 1, 1e20]
	max_acc = 0
	best_lamb = 0.00001

	for lamb in lambs:
		print('Hyperparameters Tuning ------------------>>>')
		print('Lambda = ', lamb)
		sys.stdout.flush()
		model = LogisticRegression(C=lamb,verbose=0, fit_intercept=True, max_iter=1000,class_weight='balanced')
		model.fit(X_train, Y_train)

		print('training score',model.score(X_train, Y_train))

		preds = model.predict(X_dev)
		acc = (Y_dev == preds).mean()
		print('validation accuracy = ', acc)
		sys.stdout.flush()

		if acc > max_acc:
			max_acc = acc
			best_lamb = lamb

	print("++++++++++++++ FINAL ROUND ++++++++++++")
	print("Choose lambda = ", best_lamb)
	sys.stdout.flush()
	model = LogisticRegression(C=best_lamb,verbose=0, fit_intercept=True, max_iter=1000,class_weight='balanced')
	model.fit(X_train, Y_train)


	print('Model parameters:')
	print(model.intercept_, model.coef_)

	pickle.dump(model, open(MODEL_FINAL_FILE, 'wb'))

if __name__ == '__main__':
	print(judge('doan ke thien cau giay ha noi'))