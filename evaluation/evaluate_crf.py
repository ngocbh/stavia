import sys, codecs
sys.path.append('..')

from stavia.crf.feature_extraction import extract_features
from stavia.crf.data_processing import load_data
from stavia.utils.parameters import *

from sklearn.metrics import classification_report

import pycrfsuite
import numpy as np

CRF_TEST_FILE='data/test_crf.txt'

FIELDS = ['CITY','DIST', 'WARD', 'STREET', 'PRO']
ENTITY_PREDS = {'CITY': 0, 'DIST': 0, 'WARD': 0, 'STREET': 0, 'PRO': 0}
ENTITY_TRUTHS = {'CITY': 0, 'DIST': 0, 'WARD': 0, 'STREET': 0, 'PRO': 0}
ENTITY_TRUE = {'CITY': 0, 'DIST': 0, 'WARD': 0, 'STREET': 0, 'PRO': 0}
PRECISION = {'CITY': 0, 'DIST': 0, 'WARD': 0, 'STREET': 0, 'PRO': 0}
RECALL = {'CITY': 0, 'DIST': 0, 'WARD': 0, 'STREET': 0, 'PRO': 0}
TRUE_SENT = 0
TOTAL_SENT = 0

def detect_entity(labels):
	buff = ''
	lbuff = ''
	ret = {'CITY': set(), 'DIST': set(), 'WARD': set(), 'STREET': set(), 'PRO': set()}
	isEntity = False
	for i in range(len(labels)):
		label = labels[i]
		if label[0] == 'I':
			buff += '|' + str(i) 
		else:
			if isEntity == True:
				ret[lbuff].add(buff)
			if label[0] == 'B':
				buff = str(i)
				lbuff = label[2:]
				isEntity = True
			else:
				isEntity = False
	if len(buff) > 0:
		ret[lbuff].add(buff)

	return ret



def evaluate_entity(predictions, truths):
	global TRUE_SENT, TOTAL_SENT
	for t in range(len(truths)):
		pred_entities = detect_entity(predictions[t])
		truth_entities = detect_entity(truths[t])
		isTrueSen = True
		for field in FIELDS:
			ENTITY_PREDS[field] += len(pred_entities[field])
			ENTITY_TRUTHS[field] += len(truth_entities[field])
			tp = 0
			for entity in pred_entities[field]:
				if entity in truth_entities[field]:
					tp += 1
			ENTITY_TRUE[field] += tp
			if tp != len(pred_entities[field]) or tp != len(truth_entities[field]):
				isTrueSen = False
		if isTrueSen == True:
			TRUE_SENT += 1

	TOTAL_SENT = len(truths)



def evaluate_crf():
	data = load_data(CRF_TEST_FILE)
	X_test = [extract_features(doc) for doc, labels in data]
	Y_test = [labels for doc, labels in data]

	tagger = pycrfsuite.Tagger()
	tagger.open(CRF_MODEL_FILE)

	Y_pred = [tagger.tag(xseq) for xseq in X_test]

	evaluate_entity(Y_pred, Y_test)

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

	total_true = 0
	total_truth = 0
	total_pred = 0
	# Print out the classification report
	with codecs.open('results/crf_metrics{}.txt'.format('_norat' if USE_RAT == False else '_rat'),encoding='utf8', mode='w') as f:
		f.write(classification_report(
		    truths, predictions,
		    target_names=target_names,digits=4))

		f.write('Entity evaluation' + '\n')

		f.write('ENTITY_TRUE' + '\n')
		for field in FIELDS:
			f.write(field + '=' + str(ENTITY_TRUE[field]) + '\t|\t')
			total_true += ENTITY_TRUE[field]
		f.write('\n')

		f.write('ENTITY_PREDS' + '\n')
		for field in FIELDS:
			f.write(field + '=' + str(ENTITY_PREDS[field]) + '\t|\t')
			total_pred += ENTITY_PREDS[field]
		f.write('\n')

		f.write('ENTITY_TRUTHS' + '\n')
		for field in FIELDS:
			f.write(field + '=' + str(ENTITY_TRUTHS[field]) + '\t|\t')
			total_truth += ENTITY_TRUTHS[field]
		f.write('\n')

		f.write('PARTIAL PRECISION' + '\n')
		for field in FIELDS:
			if ENTITY_PREDS[field] == 0:
				PRECISION[field] = 1.0
			else:
				PRECISION[field] = ENTITY_TRUE[field]/float(ENTITY_PREDS[field])
			f.write(field + '=' + str(PRECISION[field]) + '\t|\t')
		f.write('\n')

		f.write('PARTIAL RECALL' + '\n')
		for field in FIELDS:
			if ENTITY_TRUTHS[field] == 0:
				RECALL[field] = 1.0
			else:
				RECALL[field] = ENTITY_TRUE[field]/float(ENTITY_TRUTHS[field])
			f.write(field + '=' + str(RECALL[field]) + '\t|\t')
		f.write('\n')

		f.write('PARTIAL FSCORE' + '\n')
		for field in FIELDS:
			f.write(field + '=' + str(2*PRECISION[field]*RECALL[field] / float(PRECISION[field] + RECALL[field])) + '\t|\t')
		f.write('\n')

		total_precision = total_true/float(total_pred)
		total_recall = total_true/float(total_truth)
		f.write('average precision = ' + str(total_precision) + '\n')
		f.write('average recall = ' + str(total_recall) + '\n')
		f.write('average recall = ' + str(2*total_precision*total_recall / (total_precision + total_recall)) + '\n')
		f.write('\n')

		f.write('true sentences = ' + str(TRUE_SENT) + '\n')
		f.write('total sentences = ' + str(TOTAL_SENT) + '\n')
		f.write('sentence accuracy = ' + str(TRUE_SENT/float(TOTAL_SENT)) + '\n')

if __name__ == "__main__":
	evaluate_crf()