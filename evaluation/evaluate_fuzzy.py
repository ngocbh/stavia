import sys, codecs
sys.path.append('..')

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from stavia.feature_extraction import extract_features
from stavia.data_processing import load_data
from stavia.utils.parameters import *
from stavia.utils.utils import create_status
from stavia.utils import utils

import json
import stavia
import numpy as np
import requests
import time

TEST_FINAL_FILE='data/test_final{}_{}.json'.format('_small' if IS_BUILDING_STAGE == 1 else '', DATASET_ID)

s = requests.Session()

def match(ngram, rest, field, rest_field, query_size=100):
	query = json.dumps({
		"from": 0, "size": query_size,
		"query": {
			"bool": {
				"must": {
					"match_phrase": {
						field : ngram
					}
				},
				"should": {
					"multi_match": {
						"query": rest,
						"type": "cross_fields",
						"fields": rest_field,
						"minimum_should_match": 0
					}
				}
			}
		}
	})

	headers = {'Content-Type': 'application/json'}
	response = s.get(SEARCHING_URI, data=query, headers=headers)
	results = json.loads(response.text)
	return results["hits"]["hits"]

def get_candidate(address, field):
	#neu ko co ky tu co dau nao thi truy van tren truong khong dau???
	useVietChar = utils.contains_Vietchar(address)
	
	rest_field = [e for e in FIELDS if e != field]
	if useVietChar == False:
		field = field + '_no'
		for i in range(len(rest_field)):
			rest_field[i] = rest_field[i] + '_no'

	candidates_dict = {}
	for i in range(2,4):
		ngrams = utils.generate_ngrams_word_level(address, n=i)
		if len(ngrams) == 0:
			continue
		mini_query_size = (QUERY_SIZE-len(candidates_dict))/(len(ngrams)*(4-i))
		for ngram, rest in ngrams:
			# address = utils.no_accent_vietnamese(address)
			query_results = match(ngram, rest, field, rest_field , mini_query_size)
			for candidate in query_results:
				_id = candidate['_id']
				if _id not in candidates_dict:
					candidates_dict[_id] = candidate
				else:
					if candidates_dict[_id]['_score'] < candidate['_score']:
						candidates_dict[_id] = candidate

	candidates_list = [candidate for _id, candidate in candidates_dict.items()]
	return  candidates_list


def evaluate_fuzzy():
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
		isMatch = False
		candidates = []
		for field in FIELDS:
			candidates_temp = get_candidate(raw_add, field)
			candidates.extend(candidates_temp)
		for candidate in candidates:
			if str(candidate['_id']) in std_add:
				isMatch = True
		if isMatch == False:
			sagel_error += 1
			error_record = {}
			error_record['raw_add'] = raw_add
			error_record['std_add'] = std_add
			errors.append(error_record)

		total_sample += 1
		next(status)

	with codecs.open('results/error_fuzzy_{}.json'.format(MODEL_ID), encoding='utf8', mode='w') as f:
		jstr = json.dumps(errors,ensure_ascii=False, indent=4)
		f.write(jstr + '\n')
		f.write(str(sagel_error) +'\n')
		f.write(str(total_sample))
	print(sagel_error)
	print(total_sample)


if __name__ == "__main__":
	evaluate_fuzzy()

