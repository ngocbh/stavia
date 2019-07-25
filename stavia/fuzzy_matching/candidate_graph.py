from __future__ import absolute_import

from .node import Node, Type
from .preprocessing import preprocess_address
from .query_elasticsearch import get_candidate
from ..utils.utils import selection_sort, copy_stack
from ..utils.parameters import *

import pydot

ROOT = Node(value='việt nam', label='country', type=Type.EXPLICIT, score=0, level=0, MCS=0)

class CandidateGraph():

	def __init__(self):
		self.root = ROOT
		self.nodes = {self.root.id: self.root}
		self.edges = {}

	def __get_child_field(self, parent_level, std_addr):
		for i in range(parent_level+1,6):
			if MAP_FIELD[i] in std_addr:
				return MAP_FIELD[i]

		return None

	def __add_node(self, value, label, level, score, parent_node, field):
		_id = hash(parent_node.id ^ hash(value + label))

		if _id in self.nodes:
			if label == field:
				self.nodes[_id].score = score
				self.nodes[_id].type = Type.EXPLICIT
			return self.nodes[_id]

		node = Node(value=value, label=label, level=level, MCS=0)
		node.id = _id
		node.pid = parent_node.id

		if label == field:
			node.score = score
			node.type = Type.EXPLICIT
		else:
			node.score = 0
			node.type = Type.IMPLICIT

		self.nodes[_id] = node
		return node

	def __add_edge(self,pnode, cnode):
		if pnode.id in self.edges:
			self.edges[pnode.id].add(cnode.id)
		else:
			self.edges[pnode.id] = set([cnode.id])

	def __update_score(self, cnode):
		sum_score = 0
		while cnode.pid != None:
			if cnode.type == Type.EXPLICIT:
				sum_score += cnode.score
			cnode.MCS = max(cnode.MCS, sum_score)
			cnode = self.nodes[cnode.pid]
			
		cnode.MCS = max(cnode.MCS, sum_score)
		return sum_score

	@staticmethod
	def build_graph(addr):
		# do we really need removing duplicate n-gram???
		graph = CandidateGraph()
		addr = preprocess_address(addr) 
		for field in FIELDS:
			candidates = get_candidate(addr, field)

			for candidate in candidates:
				score = candidate['_score']
				id_addr = candidate['_id']
				std_addr = candidate['_source']

				parent_node = graph.root
				child_field = graph.__get_child_field(graph.root.level, std_addr)
				while child_field != None:
					cnode = graph.__add_node(std_addr[child_field], child_field, MAP_LEVEL[child_field], score, parent_node, field)
					graph.__add_edge(parent_node,cnode)
					child_field = graph.__get_child_field(cnode.level, std_addr)
					parent_node = cnode

				# parent_node is leaf node now
				parent_node.addr_id = id_addr
				graph.__update_score(parent_node)

		return graph

	def __prune(self, node_id):
		if node_id in self.edges:
			for child in self.edges[node_id]:
				self.__prune(child)
		self.nodes.pop(node_id, None)
		self.edges.pop(node_id, None)

	def __beam_search(self, node_id, k):
		if node_id not in self.edges:
			return

		childs = [(child_id, self.nodes[child_id].MCS)  for child_id in self.edges[node_id]]
		childs = sorted(childs, key=lambda kv:(kv[1], kv[0]))
		n = len(childs)

		for i in range(0,n):
			if i < n-k:
				self.__prune(childs[i][0])
				self.edges[node_id].remove(childs[i][0])
			else:
				self.__beam_search(childs[i][0],k)

	def prune_by_beam_search(self, k=2):
		self.__beam_search(self.root.id,k)

	def prune_by_heap(self):
		pass

	def visit_graph(self, node_id, stack, results):
		stack.append(node_id)
		childs = list(self.edges[node_id]) if node_id in self.edges else None

		if childs != None:
			for child in childs:
				self.visit_graph(child, stack, results)

		if self.nodes[node_id].addr_id != None:
			results.append(copy_stack(stack))

		stack.pop()

	def extract_address(self):
		stack = []
		results = []
		self.visit_graph(self.root.id, stack, results)
		
		ret = []
		for id_set in results:
			addr = {}
			score = 0
			for node_id in id_set:
				node = self.nodes[node_id]
				addr[node.label] = node.value
				addr[node.label + '_score'] = node.score
				score += node.score
			addr['addr_id'] = int(self.nodes[id_set[-1]].addr_id)
			ret.append(addr)
		return ret

	def display_graph(self):
		G = pydot.Dot(graph_type='digraph')
		
		for key, node in self.nodes.items():
			gnode = pydot.Node(str(node.id) + '\n' + str(node.label) + '\n' + node.value + '\n' + str(node.score),style='filled', fillcolor='yellow')
			G.add_node(gnode)

		for frid, to_set in self.edges.items():
			for toid in list(to_set):
				frnode = self.nodes[frid]
				tonode = self.nodes[toid]
				edge = pydot.Edge(str(frnode.id) + '\n' + str(frnode.label) + '\n' + frnode.value + '\n' + str(frnode.score), 
					str(tonode.id) + '\n' + str(tonode.label) + '\n' + tonode.value + '\n' + str(tonode.score))
				G.add_edge(edge)

		G.write_png('candidate_graph.png')
		print('Created candidate graph image!')


	def print_graph(self):
		print(self.nodes)
		print(self.edges)




