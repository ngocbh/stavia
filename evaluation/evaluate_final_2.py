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
def find_entities(std_add):
	std_entities = {}
	for _id, addr in std_add.items():
		for field, value in addr.items():
			std_entities[field] = value
	return std_entities

def evaluate_final():
	data = load_data(TEST_FINAL_FILE)
	print('DATASET_ID =',DATASET_ID)
	print('MODEL_ID =',MODEL_ID)
	true_sample = 0
	score = 0
	pscore = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	total_sample = 0
	sagel_error = 0


	truepos_example = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	relevant_examples = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	selected_examples = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}

	partial_precision = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	partial_recall = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	partial_f1score = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}

	sum_truepos = 0
	sum_relevant = 0
	sum_selected = 0

	average_precision = 0
	average_recall = 0
	average_f1score = 0

	errors = []
	sagel_error_list = []
	status = iter(create_status(len(data)))
	for raw_add, std_add in data:
		result, ranked_list, crf_entities, words, labels = stavia.standardize4testing(raw_add)

		if result != None and 'addr_id' in result and str(result['addr_id']) in std_add:
			true_sample += 1
			score += 3
			for field in FIELDS:
				pscore[field] += 1
			for field in FIELDS: 
				if field in result:
					truepos_example[field] += 1
					relevant_examples[field] += 1
					selected_examples[field] += 1
		else:
			if result == None or 'addr_id' not in result:
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

			error_sample = {}
			max_score = 0
			max_pscore = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
			for _id, addr in std_add.items():
				score_temp = 0
				pscore_temp = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
				for field in FIELDS:
					if field in result and addr[field] != 'None':
						if addr[field] == result[field]:
							score_temp += 1
							pscore_temp[field] += 1
							if field == 'district':
								error_sample['type'] = 'true_district'
							continue
					break
				if score_temp > max_score:
					max_score = score_temp
					max_pscore = pscore_temp
			score += max_score
			for field in FIELDS:
				pscore[field] += max_pscore[field]

			error_sample['raw_add'] = raw_add
			error_sample['result'] = result
			error_sample['std_add'] = std_add

			true_candidate = []
			for candidate in ranked_list:
				if str(candidate['addr_id']) in std_add:
					true_candidate.append(candidate)

			error_sample['true_candidate'] = true_candidate
			error_sample['crf_entities'] = crf_entities
			error_sample['crf_tag'] = [[word,label] for word, label in zip(words,labels)]
			if len(true_candidate) == 0:
				sagel_error += 1
				sagel_error_list.append(error_sample)
			errors.append(error_sample)



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

	average_precision = sum_truepos/float(sum_selected)
	average_recall = sum_truepos/float(sum_relevant)
	average_f1score = 2*average_precision*average_recall/(average_precision + average_recall)


	print(str(true_sample) + ' ' + str(total_sample) + '\n')
	print('sagel_error' + str(sagel_error))
	print('accuracy = ' + str(true_sample/float(total_sample)) +' \n')
	print('score =' + str(score) + '/' + str(total_sample*3) + '=' + str(score/float(total_sample*3)) + '\n')
	print('partial score' + '\n')
	for field in FIELDS:
		print(field + '_score =' + str(pscore[field]) + '\n')


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


	with codecs.open('results/final_result_{}.txt'.format(MODEL_ID), encoding='utf8', mode='w') as f:
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

		f.write(str(true_sample) + ' ' + str(total_sample) + '\n')
		f.write('sagel_error' + str(sagel_error) + '\n')
		f.write('accuracy = ' + str(true_sample/float(total_sample)) +' \n')
		f.write('score =' + str(score) + '/' + str(total_sample*3) + '=' + str(score/float(total_sample*3)) + '\n')
		f.write('partial score' + '\n')
		for field in FIELDS:
			f.write(field + '_score =' + str(pscore[field]) + '\n')


	with codecs.open('results/error_pattern_final_{}.json'.format(MODEL_ID), encoding='utf8', mode='w') as f:
		js_data = {'error': errors}
		jstr = json.dumps(js_data,ensure_ascii=False, indent=4)
		f.write(jstr)
	
	with codecs.open('results/error_sagel_final_{}.json'.format(MODEL_ID), encoding='utf8', mode='w') as f:
		js_data = {'error': sagel_error_list}
		jstr = json.dumps(js_data,ensure_ascii=False, indent=4)
		f.write(jstr)

if __name__ == "__main__":
	evaluate_final()

