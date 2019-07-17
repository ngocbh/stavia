from __future__ import absolute_import

from .node import Node, Type
from .preprocessing import preprocess_address
from .query_elasticsearch import get_candidate

FIELDS =['city', 'district', 'ward', 'street']
MAP_LEVEL = {'country': 0, 'city': 1, 'district': 2, 'ward': 3, 'street': 4, 'name': 5}
MAP_FIELD = {0: 'country', 1: 'city', 2:'district', 3: 'ward', 4: 'street', 5: 'name'}

ROOT = Node(value='Việt Nam', label='country', type=Type.EXPLICIT, score=0, level=0, MCS=0)

class CandidateGraph():

	def __init__(self):
		self.root = ROOT
		self.nodes = {ROOT.id: ROOT}
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

		return sum_score

	@staticmethod
	def build_graph(addr):
		# do we really need removing duplicate n-gram???
		graph = CandidateGraph()
		addr = preprocess_address(addr) 
		for field in FIELDS:
			candidates = get_candidate(addr, field)

			print(len(candidates))

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

				graph.__update_score(leaf=parent_node)

		return graph

	def printGraph(self):
		print(self.nodes)
		print(self.edges)




