from __future__ import absolute_import

import json
import requests
import os
from ..utils import utils
from ..utils.parameters import *

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

