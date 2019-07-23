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

IS_BUILDING_STAGE = 0

TEST_FINAL_FILE='data/test_final{}.json'.format('_small' if IS_BUILDING_STAGE == 1 else '')


def evaluate_final():
	data = load_data(TEST_FINAL_FILE)
	true_sample = 0
	total_sample = 0

	errors = []

	status = iter(create_status(len(data)))
	for raw_add, std_add in data:
		result = stavia.standardize(raw_add)
		if result == None:
			print('result error')
		if int(result['addr_id']) == int(std_add['id']):
			true_sample += 1
		else:
			error_sample = {}
			error_sample['raw_add'] = raw_add
			error_sample['result'] = result
			error_sample['std_add'] = std_add
			errors.append(error_sample)

		total_sample += 1
		next(status)

	with codecs.open('results/final_result.txt', encoding='utf8', mode='w') as f:
		f.write(str(true_sample) + ' ' + str(total_sample) + '\n')
		f.write('accuracy = ' + str(true_sample/float(total_sample)) +' \n')

	with codecs.open('results/error_pattern_final.json', encoding='utf8', mode='w') as f:
		js_data = {'error': errors}
		jstr = json.dumps(js_data,ensure_ascii=False, indent=4)
		f.write(jstr)
	

if __name__ == "__main__":
	evaluate_final()

