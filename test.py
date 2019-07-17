from stavia.fuzzy_matching import query_elasticsearch
from stavia.fuzzy_matching.candidate_graph import CandidateGraph
from stavia.fuzzy_matching.node import Node

graph = CandidateGraph.build_graph('nguyễn thanh cần tân an long an')
graph.printGraph()