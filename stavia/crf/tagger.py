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

