from __future__ import absolute_import

from .utils.parameters import *
from .utils.utils import contains_Vietchar, no_accent_vietnamese, tokenize
from nltk import ngrams

import re

import numpy as np
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

def tokenize_field(name, field):
	if field == 'district':
		field = 'DIST'
	else:
		field = field.upper()
	words = tokenize(name)
	labels = []
	if (len(words)) != 0:
		labels.append('B_' + field)
	for _ in range(len(words)-1):
		labels.append('I_' + field)
	return words, labels


def extract_features(raw_add, words, labels, entities, candidate):
	features = {}
	#Bias
	features.update({'bias': 1})
	#Lexical feature
	if USE_LEXICAL_FEATURES == True:
		for field in FIELDS:
			if field in candidate.keys():
				cdd_words, cdd_labels = tokenize_field(candidate[field], field)
				for cdd_word, cdd_label in zip(cdd_words, cdd_labels):
					for word, label in zip(words, labels):
						if label == cdd_label:
							features.update({'cdd:{}:{}:{}'.format(cdd_label,cdd_word,word.lower()): 1})

	#Admin_level in crf
	for field in FIELDS:
		if field in entities:
			features.update({'crf:{}:lv'.format(field) :  1})

	#Admin_level in candidate
	for field in FIELDS:
		if field in candidate.keys():
			features.update({'cdd:{}:lv'.format(field) : 1})

	#Is contain vietnamese character
	if contains_Vietchar(raw_add) == True:
		features.update({'isVietnamese': 1})

	#Elastic Score
	for field in FIELDS:
		if field + '_score' in candidate.keys():
			features.update({'el:{}:s'.format(field) : float(candidate[field+'_score'])} )

	#Entity Score
	for field in FIELDS:
		if field in entities and field in candidate.keys():
			value = 0
			for entity in entities[field]:
				value = max(value, 1 if entity.lower() == candidate[field].lower() else 0)
			features.update({'en:{}:s'.format(field): value})

	#Entity Score with no_accent_vietnamese
	for field in FIELDS:
		if field in entities and field in candidate.keys():
			value = 0
			for entity in entities[field]:
				value = max(value, 1 if no_accent_vietnamese(entity.lower()) == no_accent_vietnamese(candidate[field].lower()) else 0)
			features.update({'e:{}:sn'.format(field): value})

	#Jaccard Score
	for field in FIELDS:
		if field in entities and field in candidate.keys():
			value = 0
			for entity in entities[field]:
				value = max(value,jaccard_similarity(entity, candidate[field]))
			features.update({'j:{}:s'.format(field): value})

	#Jaccard Score with no_accent_vietnamese
	for field in FIELDS:
		if field in entities and field in candidate.keys():
			value = 0
			for entity in entities[field]:
				value = max(value, jaccard_similarity(no_accent_vietnamese(entity.lower()), no_accent_vietnamese(candidate[field].lower())))
			features.update({'j:{}:sn'.format(field): value})

	#Levenshtein Score
	for field in FIELDS:
		if field in entities and field in candidate.keys():
			for entity in entities[field]:
				value = max(value, levenshtein_ratio_and_distance(entity.lower(), candidate[field].lower()))
			features.update({'l:{}:s'.format(field): value})

	#Levenshtein Score with no_accent_vietnamese
	for field in FIELDS:
		if field in entities and field in candidate.keys():
			value = 0
			for entity in entities[field]:
				value = max(value, levenshtein_ratio_and_distance(no_accent_vietnamese(entity.lower()), no_accent_vietnamese(candidate[field].lower())))
			features.update({'l:{}:sn'.format(field): value})

	return features