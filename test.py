from stavia.fuzzy_matching import query_elasticsearch
from stavia.fuzzy_matching.candidate_graph import CandidateGraph
from stavia.fuzzy_matching.node import Node
from stavia.data_processing import load_data
from stavia.fuzzy_matching import sagel
from stavia.utils.utils import create_status
from stavia.utils.parameters import *
from stavia.crf import crf_based_standardization as cbs
import stavia

# graph = CandidateGraph.build_graph('Số 27 , ngõ 594 Đường Láng , Đống Đa , Hà Nội')
# graph.prune_by_beam_search(k=3)
# graph.display_graph()
# print(graph.extract_address())
# result = sagel.get_sagel_answer(graph)
# print(result)

# print(stavia.standardize('Số 27 , ngõ 594 Đường Láng , Đống Đa , Hà Nội'))

print(cbs.standardize('Số 27 , ngõ 594 Đường Láng , Đống Đa , Hà Nội'))