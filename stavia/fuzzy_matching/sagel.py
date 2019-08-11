from __future__ import absolute_import

from .node import Node, Type
from .candidate_graph import CandidateGraph
from ..utils.utils import selection_sort, copy_stack

def visit_graph_sagel(graph, node, stack, result):
	stack.append(node.id)
	childs = list(graph.edges[node.id]) if node.id in graph.edges else None

	if childs != None:
		map = {}
		for child_id in childs:
			sum = graph.nodes[child_id].MCS
			map.update({child_id: sum})
		selection_sort(map, childs)
		for child_id in childs:
			return visit_graph_sagel(graph, graph.nodes[child_id], stack, result)

	# if node.type == Type.EXPLICIT:
	stack_copy = copy_stack(stack)
	result.append(stack_copy)

	stack.pop()
	return stack, result

def get_sagel_answer(graph):
	# print(self.root.MCS)
	stack = []
	result = []

	stack, result = visit_graph_sagel(graph, graph.root, stack, result)
	if len(result) == 0:
		return None
		
	ret = {'score': graph.root.MCS} 
	for _id in result[0]:
		node = graph.nodes[_id]
		if node.addr_id != None:
			ret['addr_id'] = node.addr_id 
		ret[node.label] = node.value
		ret[node.label + '_score'] = node.score

	return ret