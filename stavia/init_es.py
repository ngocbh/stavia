from __future__ import absolute_import

from .utils.parameters import *

import requests, codecs
import json

def delete_exist_package():
	response = requests.delete(URI)
	# if response.status_code == 404:
	# 	raise Exception("Error when deleting exist index")

def put_configuration():
	with open(CONFIGURATION_FILE) as f:
		conf = json.load(f)
	headers = {"Content-Type": "application/x-ndjson"}

	response = requests.put(URI, headers=headers, data=conf)
	if response.status_code == 404:
		raise Exception("Error when configuring index")

def post_data():
	with codecs.open(RAT_FILE, encoding='utf8', mode='r') as ratf:
		rat_data = ratf.read()
	data_to_post = rat_data.encode('utf8')
	headers = {"Content-Type": "application/x-ndjson"}
	response = requests.post(URI+'doc/_bulk?pretty', data=data_to_post, headers=headers)
	if response.status_code == 404:
		raise Exception("Error when posting data")


def init_es():
	response = requests.head(URI)
	if response.status_code == 200:
		print('ElasticSearch has been initialized before, skip init!')
		return

	delete_exist_package()
	put_configuration()
	post_data()
	print('Initialized')

def reinit_es():
	delete_exist_package()
	put_configuration()
	post_data()
	print('Re-Initialized')

if __name__ == '__main__':
	reinit_es()