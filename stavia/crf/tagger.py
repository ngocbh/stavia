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

def detect_entity(inp):
	tokens, labels = tag(inp)
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

