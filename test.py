
from stavia.fuzzy_matching import query_elasticsearch
from stavia.fuzzy_matching.candidate_graph import CandidateGraph
from stavia.fuzzy_matching.node import Node
from stavia.fuzzy_matching import sagel

from stavia.crf.trainner import train_crf
from stavia.crf.tagger import tag, detect_entity
from stavia.init_es import init_es


init_es()
graph = CandidateGraph.build_graph('bệnh viện bách khoa bạch mai hai ba trung hoan kiem')
sagel_result = sagel.get_sagel_answer(graph)

print(sagel_result)

graph.prune_by_beam_search(k=3)
graph.display_graph()

print(graph.extract_address())
print(detect_entity('bệnh viện bách khoa bạch mai hai ba trung hoan kiem'))