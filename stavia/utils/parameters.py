from __future__ import absolute_import

import os
WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
# ITERMEDIATE
IS_BUILDING_STAGE=0
VERBOSE=1
METHOD='llm' #llm for log-linear model and lr for logistic regression

#GLOBAL
PUNCTUATIONS=',.-()#|/\\'

#ELASTICSEARCH
HOST = "http://localhost:9200/"
try:
    HOST = os.environ['ELASTICSEARCH_HOST']
except:
    pass

URI = HOST+"smart_address"
SEARCHING_URI = HOST+"smart_address/_search"
CONFIGURATION_FILE=os.path.join(WORKING_DIR, '_data/configuration.json')
RAT_FILE=os.path.join(WORKING_DIR, '_data/rat_data.json')
FUZZINESS=1
QUERY_SIZE=2000
SLOP=2

#CANDIDATE GRAPH
FIELDS=['city', 'district', 'ward', 'street']
MAP_LEVEL={'country': 0, 'city': 1, 'district': 2, 'ward': 3, 'street': 4, 'name': 5}
MAP_FIELD={0: 'country', 1: 'city', 2:'district', 3: 'ward', 4: 'street', 5: 'name'}
BEAM_SIZE=5

#CRF
CRF_TRAIN_FILE=os.path.join(WORKING_DIR, '_data/train_crf{}.txt'.format('_small' if IS_BUILDING_STAGE == 1 else ''))
USE_RAT=False
CRF_MODEL_FILE=os.path.join(WORKING_DIR, '_data/crf{}.model'.format('_norat' if USE_RAT == False else '_rat'))
RAT_DICT_FILE=os.path.join(WORKING_DIR, '_data/rat_dict.json')

#RERANKING
TRAIN_FINAL_FILE=os.path.join(WORKING_DIR, '_data/train_final{}.json'.format('_small' if IS_BUILDING_STAGE == 1 else ''))
MODEL_FINAL_FILE=os.path.join(WORKING_DIR, '_data/final_{}.model'.format(METHOD))


