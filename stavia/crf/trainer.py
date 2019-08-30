from __future__ import absolute_import

from ..utils.parameters import *
from .feature_extraction import extract_features
from .data_processing import load_data

import codecs
import pycrfsuite

def train_crf():
	print('Loading data')
	data = load_data(CRF_TRAIN_FILE)
	print('Number of docs = ', len(data))
	print('Extracting Features')
	X = [extract_features(doc) for doc, labels in data]
	Y = [labels for doc, labels in data]

	trainer = pycrfsuite.Trainer(verbose=True)
	for xseq, yseq in zip(X, Y):
		trainer.append(xseq, yseq)
	
	trainer.set_params({
	    'c1': 0.1,
	    'c2': 0.01,  
	    'max_iterations': 500,
	    'feature.possible_transitions': True
	})

	print('Training')
	trainer.train(CRF_MODEL_FILE)
	print('Done')



if __name__ == "__main__":
	train_crf()