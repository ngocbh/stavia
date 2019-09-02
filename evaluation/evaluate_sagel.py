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

IS_BUILDING_STAGE = 0

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

def evaluate_sagel():
	data = load_data(TEST_FINAL_FILE)
	data = preprocess(data)
	
	true_sample = 0
	score = 0
	pscore = {'city': 0, 'district': 0, 'ward': 0, 'street': 0}
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
	MIN_NGRAMS=1
	MAX_NGRAMS=4

	status = iter(create_status(len(data)))
	for raw_add, std_add in data:
		graph = CandidateGraph.build_graph(raw_add)

		result = sagel.get_sagel_answer(graph)

		if result != None and 'addr_id' in result and str(result['addr_id']) in std_add:
			true_sample += 1
			score += 3
			for field in FIELDS:
				pscore[field] += 1
			std_entities = find_entities(std_add)
			for field in FIELDS: 
				if field in std_entities:
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

	for field in FIELDS:
		if selected_examples[field] == 0:
			partial_precision[field] = 1
		else:
			partial_precision[field] = truepos_example[field]/float(selected_examples[field])
		if relevant_examples[field] == 0:
			partial_recall[field] = 1
		else:
			partial_recall[field] = truepos_example[field]/float(relevant_examples[field])
		if partial_precision[field] == 0 and partial_recall[field] == 0:
			partial_f1score[field] = 0
		else:
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


	with codecs.open('results/sagel_result_{}.txt'.format(MODEL_ID), encoding='utf8', mode='w') as f:
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
		f.write('accuracy = ' + str(true_sample/float(total_sample)) +' \n')
		f.write('score =' + str(score) + '/' + str(total_sample*3) + '=' + str(score/float(total_sample*3)) + '\n')
		f.write('partial score' + '\n')
		for field in FIELDS:
			f.write(field + '_score =' + str(pscore[field]) + '\n')

	with codecs.open('results/error_pattern_sagel_{}.json'.format(MODEL_ID), encoding='utf8', mode='w') as f:
		js_data = {'error': errors}
		jstr = json.dumps(js_data,ensure_ascii=False, indent=4)
		f.write(jstr)
	



if __name__ == "__main__":
	evaluate_sagel()