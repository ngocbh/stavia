from stavia.fuzzy_matching import query_elasticsearch
from stavia.fuzzy_matching.candidate_graph import CandidateGraph
from stavia.fuzzy_matching.node import Node
from stavia.data_processing import load_data
from stavia.fuzzy_matching import sagel
from stavia.utils.utils import create_status
from stavia.utils.parameters import *

graph = CandidateGraph.build_graph('Dự án Mỹ Thái 1 , Quận 7 , HỒ Chí Minh')
result = sagel.get_sagel_answer(graph)
print(result)
# print(stavia.standardize(u'Phố Hàng Chiếu , Hà Nội'))
