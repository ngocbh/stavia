from __future__ import absolute_import

import codecs, json

def load_data(filename):
	data = []
	with codecs.open(filename, encoding='utf8', mode='r') as f:
		sample_list = json.load(f)['data']
		for sample in sample_list:
			data.append((sample['noisy_add'],sample['std_add']))
	return data
