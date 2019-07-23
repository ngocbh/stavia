import sys, codecs
sys.path.append('..')

from stavia.fuzzy_matching import query_elasticsearch
from stavia.fuzzy_matching.candidate_graph import CandidateGraph
from stavia.fuzzy_matching.node import Node
from stavia.fuzzy_matching import sagel
from stavia.utils import create_status
import json

IS_BUILDING_STAGE = 1

def evaluate_sagel():
	error_pattern = []
	with codecs.open('data/testset{}.json'.format('_mini' if IS_BUILDING_STAGE == 1 else ''), encoding='utf8', mode='r') as tf:
		data = json.load(tf)['data']

		no_true = 0
		no_pattern = 0

		status = iter(create_status(len(data)))
		for pattern in data:
			no_pattern += 1

			graph = CandidateGraph.build_graph(pattern['noisy_add'])
			sagel_result = sagel.get_sagel_answer(graph)

			try: 
				if int(sagel_result['addr_id']) == int(pattern['std_add']['id']):
					no_true += 1
			except:
				pattern['sagel_result'] = sagel_result
				error_pattern.append(pattern)
			next(status)

	with codecs.open('results/error_pattern_sagel.json', encoding='utf8', mode='w') as f:
		js_data = {'error': error_pattern}
		jstr = json.dumps(js_data,ensure_ascii=False, indent=4)
		f.write(jstr)


if __name__ == "__main__":
	evaluate_sagel()