from __future__ import absolute_import

from .tagger import tag, detect_entity
from ..utils.parameters import *
from ..utils import utils
import pandas
import random

RAT_DF = pandas.read_csv(RAT_FILE_CSV)
RAT_DF_NV = pandas.read_csv(RAT_NV_FILE_CSV)

def normalize(name):
	return name.lower()

def match_std_add(entities, rat_df):
	std_add = {}
	
	res_df = rat_df
	fields = ['street', 'ward', 'district', 'city']

	is2record = False
	if 'ward' in entities and 'street' in entities:
		res_df = res_df.loc[(res_df['street'] == entities['street']) | (res_df['ward'] == entities['ward'])]
		is2record = True
		for i in range(2,4):
			field = fields[i]
			if field not in entities:
				continue
			res_df = res_df.loc[res_df[field] == entities[field]]
	elif 'ward' not in entities and 'street' not in entities:
		res_df = res_df.loc[(res_df['street'] == 'None') & (res_df['ward'] == 'None')]
		for i in range(2,4):
			field = fields[i]
			if field not in entities:
				continue
			res_df = res_df.loc[res_df[field] == entities[field]]
	else:
		for field in fields:
			if field not in entities:
				continue
			res_df = res_df.loc[res_df[field] == entities[field]]

	if res_df.shape[0] == 0:
		return None
	_id = int(res_df.iloc[0]['id'])
	std_add = {}
	std_add['id'] = _id
	if res_df.iloc[0]['street'] != 'None':
		std_add['street'] = res_df.iloc[0]['street']
	if res_df.iloc[0]['ward'] != 'None':
		std_add['ward'] = res_df.iloc[0]['ward']
	if res_df.iloc[0]['district'] != 'None':
		std_add['district'] = res_df.iloc[0]['district']
	if res_df.iloc[0]['city'] != 'None':
		std_add['city'] = res_df.iloc[0]['city']
		return std_add

def standardize(addr):
	useVietChar = utils.contains_Vietchar(addr)
	ret = None
	entities_list = detect_entity(addr)
	entities_dict = {}
	for entity_name, field, loc in entities_list:
		entities_dict[field] = normalize(entity_name)
	if useVietChar:
		ret = match_std_add(entities_dict, RAT_DF)
	else:
		ret = match_std_add(entities_dict, RAT_DF_NV)
	return ret


if __name__ == '__main__':
	standardize('Số 27 , ngõ 594 Đường Láng , Đống Đa , Hà Nội')