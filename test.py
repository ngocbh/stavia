
'''
from stavia.fuzzy_matching import query_elasticsearch
from stavia.fuzzy_matching.candidate_graph import CandidateGraph
from stavia.fuzzy_matching.node import Node
from stavia.fuzzy_matching import sagel

graph = CandidateGraph.build_graph('Hưng Gia 2 - Phú Mỹ Hưng , Quận 7 , Hồ Chí Minh')
sagel_result = sagel.get_sagel_answer(graph)

print(sagel_result)

graph.prune_by_beam_search()
graph.display_graph()

print(graph.extract_address())
'''

from stavia.crf.trainner import train_crf
from stavia.crf.tagger import tag

print(tag('toi muon mua hai ba trung hoan kiem ha noi'))