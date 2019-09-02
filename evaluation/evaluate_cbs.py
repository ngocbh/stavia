import sys, codecs
sys.path.append('..')

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from stavia.feature_extraction import extract_features
from stavia.data_processing import load_data
from stavia.utils.parameters import *
from stavia.utils.utils import create_status

import json
import stavia
import numpy as np

TEST_FINAL_FILE='data/test_final{}_{}.json'.format('_small' if IS_BUILDING_STAGE == 1 else '', DATASET_ID)

# TEST_FINAL_FILE='data/test_final_small_1.json'

def preprocess(raw_data):
	ret_data = []
	for raw_add, std_add in raw_data:
		if len(std_add) == 0:
			continue
		elif len(std_add) == 1:
			ret_data.append((raw_add, std_add))
		else:
			best_add_lv = -1
			best_add = {}
			best_id = -1
			for _id, add in std_add.items():
				add_lv = 0
				for key, value in add.items():
					if key in MAP_LEVEL and value != 'None':
						add_lv = max(add_lv, MAP_LEVEL[key])

				if add_lv > best_add_lv:
					best_add_lv = add_lv
					best_add = add
					best_id = _id

			if best_add_lv != -1:
				new_std_add = {str(best_id): best_add}
				ret_data.append((raw_add, new_std_add))
	return ret_data

def find_entities(std_add):
	std_entities = {}
	for _id, addr in std_add.items():
		for field, value in addr.items():
			std_entities[field] = value
	return std_entities

def evaluate_final():
	data = load_data(TEST_FINAL_FILE)
	data = preprocess(data)
	
	print('DATASET_ID =',DATASET_ID)
	print('MODEL_ID =',MODEL_ID)
	true_sample = 0
	total_sample = 0

	truepos_example = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	relevant_examples = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	selected_examples = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}

	partial_precision = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	partial_recall = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	partial_f1score = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}

	sum_truepos = 0
	sum_relevant = 0
	sum_selected = 0

	sum_precision = 0
	sum_recall = 0
	sum_f1score = 0

	num_field = 0

	average_precision = 0
	average_recall = 0
	average_f1score = 0
	
	errors = []

	status = iter(create_status(len(data)))
	for raw_add, std_add in data:
		result = stavia.cbs.standardize(raw_add)
		if result != None and str(result['id']) in std_add:
			true_sample += 1
			std_entities = find_entities(std_add)
			for field in FIELDS: 
				if field in std_entities:
					truepos_example[field] += 1
					relevant_examples[field] += 1
					selected_examples[field] += 1
		else:
			if result == None:
				result = {}
			std_entities = find_entities(std_add)

			for field in FIELDS:
				if field in result and result[field] != 'None':
					selected_examples[field] += 1

			for field in FIELDS:
				if field in std_entities and std_entities[field] != 'None':
					relevant_examples[field] += 1

			for field in FIELDS:
				if field in result and result[field] != 'None': 
					if field in std_entities and std_entities[field] != 'None':
						if result[field] == std_entities[field]:
							truepos_example[field] += 1
						else:
							break

		total_sample += 1
		next(status)

	for field in FIELDS:
		if selected_examples[field] == 0:
			partial_precision[field] = 1
		else:
			partial_precision[field] = truepos_example[field]/float(selected_examples[field])
		if relevant_examples[field] == 0:
			partial_recall[field] = 1
		else:
			partial_recall[field] = truepos_example[field]/float(relevant_examples[field])
		partial_f1score[field] = 2*partial_precision[field]*partial_recall[field]/(partial_precision[field]+partial_recall[field])

		sum_truepos += truepos_example[field]
		sum_selected += selected_examples[field]
		sum_relevant += relevant_examples[field]

		sum_precision += partial_precision[field]
		sum_recall +=  partial_recall[field]
		sum_f1score += partial_f1score[field]
		num_field += 1

	average_precision = sum_precision/float(num_field)
	average_recall = sum_recall/float(num_field)
	average_f1score = sum_f1score/float(num_field)


	# average_precision = sum_truepos/float(sum_selected)
	# average_recall = sum_truepos/float(sum_relevant)
	# average_f1score = 2*average_precision*average_recall/(average_precision + average_recall)

	print(str(true_sample) + ' ' + str(total_sample) + '\n')
	print('accuracy = ' + str(true_sample/float(total_sample)) +' \n')
	print('partial metrics\ttruepos\tselected\trelevant\tpricision\trecall\tf1score\n')
	for field in FIELDS:
		print('{}_score\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(field, 
			str(truepos_example[field]), str(selected_examples[field]), str(relevant_examples[field]),
			str(partial_precision[field]), str(partial_recall[field]), str(partial_f1score[field]) ))

	print('average metrics\ttruepos\tselected\trelevant\tpricision\trecall\tf1score\n')
	print('average\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sum_truepos, sum_selected, sum_relevant,
		average_precision, average_recall, average_f1score))

	with codecs.open('results/cbs_result_{}.txt'.format(MODEL_ID), encoding='utf8', mode='w') as f:
		f.write(str(true_sample) + ' ' + str(total_sample) + '\n')
		f.write('accuracy = ' + str(true_sample/float(total_sample)) +' \n')
		f.write('partial metrics\ttruepos\tselected\trelevant\tpricision\trecall\tf1score\n')
		for field in FIELDS:
			f.write('{}_score\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(field, 
				str(truepos_example[field]), str(selected_examples[field]), str(relevant_examples[field]),
				str(partial_precision[field]), str(partial_recall[field]), str(partial_f1score[field]) ))

		f.write('average metrics\ttruepos\tselected\trelevant\tpricision\trecall\tf1score\n')
		f.write('average\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(sum_truepos, sum_selected, sum_relevant,
			average_precision, average_recall, average_f1score))

if __name__ == "__main__":
	evaluate_final()

