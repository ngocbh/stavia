import sys, codecs
sys.path.append('..')

from stavia.fuzzy_matching import query_elasticsearch
from stavia.fuzzy_matching.candidate_graph import CandidateGraph
from stavia.fuzzy_matching.node import Node
from stavia.data_processing import load_data
from stavia.fuzzy_matching import sagel
from stavia.utils.utils import create_status
from stavia.utils.parameters import *

import json

TEST_FINAL_FILE='data/test_final{}_{}.json'.format('_small' if IS_BUILDING_STAGE == 1 else '', DATASET_ID)

IS_BUILDING_STAGE = 1

def evaluate_sagel():
	data = load_data(TEST_FINAL_FILE)
	true_sample = 0
	score = 0
	pscore = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
	total_sample = 0

	errors = []

	status = iter(create_status(len(data)))
	for raw_add, std_add in data:
		graph = CandidateGraph.build_graph(raw_add)

		result = sagel.get_sagel_answer(graph)
		if result == None:
			print('result error')
		if str(result['addr_id']) in std_add:
			true_sample += 1
			score += 3
			for field in FIELDS:
				pscore[field] += 1
		else:
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
							continue
					break
				if score_temp > max_score:
					max_score = score_temp
					max_pscore = pscore_temp
			score += max_score
			for field in FIELDS:
				pscore[field] += max_pscore[field]

			error_sample = {}
			error_sample['raw_add'] = raw_add
			error_sample['result'] = result
			error_sample['std_add'] = std_add
			errors.append(error_sample)

		total_sample += 1
		next(status)

	with codecs.open('results/sagel_result.txt', encoding='utf8', mode='w') as f:
		f.write(str(true_sample) + ' ' + str(total_sample) + '\n')
		f.write('accuracy = ' + str(true_sample/float(total_sample)) +' \n')
		f.write('score =' + str(score) + '/' + str(total_sample*3) + '=' + str(score/float(total_sample*3)) + '\n')
		f.write('partial score' + '\n')
		for field in FIELDS:
			f.write(field + '_score =' + str(pscore[field]) + '\n')

	with codecs.open('results/error_pattern_sagel.json', encoding='utf8', mode='w') as f:
		js_data = {'error': errors}
		jstr = json.dumps(js_data,ensure_ascii=False, indent=4)
		f.write(jstr)
	



if __name__ == "__main__":
	evaluate_sagel()