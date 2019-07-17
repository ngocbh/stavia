from __future__ import absolute_import

import json
import requests
import os
from .. import utils

HOST = "http://localhost:9200/"
try:
    HOST = os.environ['ELASTICSEARCH_HOST']
except:
    pass

URI = HOST+"smart_address/_search"


def match(address, field):
    query = json.dumps({
        "from": 0, "size": 3000,
        "query": {
                "bool": {
                  "must": {
                    "match": {
                      field: {
                        "query": address,
                        "fuzziness": 1
                      }
                    }
                  },
                  "should": {
                    "match_phrase": {
                      field: {
                        "query": address,
                        "slop":  2
                      }
                    }
                  }
                }
      }
    })

    headers = {'Content-Type': 'application/json'}
    response = requests.get(URI, data=query, headers=headers)
    results = json.loads(response.text)
    return results["hits"]["hits"]

def get_candidate(address, field):
    #neu ko co ky tu co dau nao thi truy van tren truong khong dau???
    if utils.contains_Vietchar(address) == False:
        field = field +"_no"
    candidate = match(address, field)
    return  candidate
