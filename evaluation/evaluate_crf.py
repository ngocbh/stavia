import sys, codecs
sys.path.append('..')

from stavia.crf.feature_extraction import extract_features
from stavia.crf.data_processing import load_data
from stavia.utils.parameters import *

from sklearn.metrics import classification_report

import pycrfsuite
import numpy as np

CRF_TEST_FILE='data/test_crf.txt'

def evaluate_crf():
	data = load_data(CRF_TEST_FILE)
	X_test = [extract_features(doc) for doc, labels in data]
	Y_test = [labels for doc, labels in data]

	tagger = pycrfsuite.Tagger()
	tagger.open(CRF_MODEL_FILE)

	Y_pred = [tagger.tag(xseq) for xseq in X_test]

	labels = {}
	target_names = []
	num_labels = 0
	for row in Y_test:
		for label in row:
			if label not in labels:
				labels[label] = num_labels
				target_names.append(label)
				num_labels += 1

	predictions = np.array([labels[tag] for row in Y_pred for tag in row])
	truths = np.array([labels[tag] for row in Y_test for tag in row])

	# Print out the classification report
	with codecs.open('results/crf_metrics{}.txt'.format('_norat' if USE_RAT == False else '_rat'),encoding='utf8', mode='w') as f:
		f.write(classification_report(
		    truths, predictions,
		    target_names=target_names,digits=4))

if __name__ == "__main__":
	evaluate_crf()