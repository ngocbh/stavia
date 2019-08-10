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

IS_BUILDING_STAGE = 1

TEST_FINAL_FILE='data/test_final{}_{}.json'.format('_small' if IS_BUILDING_STAGE == 1 else '', DATASET_ID)


def evaluate_final():
	data = load_data(TEST_FINAL_FILE)
	print('DATASET_ID =',DATASET_ID)
	print('MODEL_ID =',MODEL_ID)
	true_sample = 0
	score = 0
	pscore = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	total_sample = 0
	sagel_error = 0

	errors = []
	status = iter(create_status(len(data)))
	for raw_add, std_add in data:
		result, ranked_list, crf_entities, words, labels = stavia.standardize4testing(raw_add)
		if result == None:
			print('result error')
		if str(result['addr_id']) in std_add:
			true_sample += 1
			score += 3
			for field in FIELDS:
				pscore[field] += 1
		else:
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
			errors.append(error_sample)



		total_sample += 1
		next(status)

	print(str(true_sample) + ' ' + str(total_sample) + '\n')
	print('sagel_error' + str(sagel_error))
	print('accuracy = ' + str(true_sample/float(total_sample)) +' \n')
	print('score =' + str(score) + '/' + str(total_sample*3) + '=' + str(score/float(total_sample*3)) + '\n')
	print('partial score' + '\n')
	for field in FIELDS:
		print(field + '_score =' + str(pscore[field]) + '\n')

	with codecs.open('results/final_result_{}.txt'.format(MODEL_ID), encoding='utf8', mode='w') as f:
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
	

if __name__ == "__main__":
	evaluate_final()

