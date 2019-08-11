from __future__ import absolute_import

from .data_processing import tokenize, wrap_postag
from .feature_extraction import extract_features
from ..utils.parameters import *

import pycrfsuite

def tag(inp):
	words = tokenize(inp)
	doc = wrap_postag(words)

	x = extract_features(doc)
	tagger = pycrfsuite.Tagger()
	tagger.open(CRF_MODEL_FILE)

	return words,tagger.tag(x)

def detect_entity(inp, tokens=None, labels=None):
	if tokens == None or labels == None:
		tokens, labels = tag(inp)
	entities = []

	n = len(tokens)
	buff = ''
	lbuff = ''
	isEntity = False
	loc = 0

	for i in range(n):
		if (labels[i][0] == 'I'):
			buff += ' ' + tokens[i]
		else:
			if isEntity == True:
				key = lbuff.lower() 
				entities.append((buff, key, str(loc)))
				loc += 1

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
		entities.append((buff, key, str(loc)))
		loc += 1


	return entities

