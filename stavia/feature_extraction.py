from __future__ import absolute_import

from .utils.parameters import *
from nltk import ngrams

import re

def get_ngrams(text, n):
    n_grams = ngrams(text, n)
    return [''.join(grams) for grams in n_grams]



def jaccard_similarity(string1, string2):
    sum = 0
    n_gram = 4
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
			features.append(jaccard_similarity(entities[field], candidate[field]))

	return features
