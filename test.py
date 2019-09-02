from stavia.fuzzy_matching import query_elasticsearch
from stavia.fuzzy_matching.candidate_graph import CandidateGraph
from stavia.fuzzy_matching.node import Node
from stavia.data_processing import load_data
from stavia.fuzzy_matching import sagel
from stavia.utils.utils import create_status
from stavia.utils.parameters import *
from stavia.crf import crf_based_standardization as cbs
from stavia.crf import tagger
import stavia

crf_entities = tagger.detect_entity('phường vân giang ninh bình ninh bình')
print(crf_entities)

graph = CandidateGraph.build_graph('phường vân giang ninh bình ninh bình')
result = sagel.get_sagel_answer(graph)
print(result)

print(stavia.standardize('vinh lai phu tho'))

print(cbs.standardize('vinh lai phu tho'))