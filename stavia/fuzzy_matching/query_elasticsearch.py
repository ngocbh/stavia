from __future__ import absolute_import

import json
import requests
import os
from ..utils import utils
from ..utils.parameters import *

def match(address, field):
	query = json.dumps({
		"from": 0, "size": QUERY_SIZE,
		"query": {
				"bool": {
				  "must": {
					"match": {
					  field: {
						"query": address,
						"fuzziness": FUZZINESS
					  }
					}
				  },
				  "should": {
					"match_phrase": {
					  field: {
						"query": address,
						"slop":  SLOP
					  }
					}
				  }
				}
	  }
	})

	headers = {'Content-Type': 'application/json'}
	response = requests.get(SEARCHING_URI, data=query, headers=headers)
	results = json.loads(response.text)
	return results["hits"]["hits"]

def get_candidate(address, field):
	#neu ko co ky tu co dau nao thi truy van tren truong khong dau???
	if utils.contains_Vietchar(address) == False:
		field = field + '_no'
	# address = utils.no_accent_vietnamese(address)
	candidate = match(address, field)
	return  candidate
