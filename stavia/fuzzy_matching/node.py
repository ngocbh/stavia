from __future__ import absolute_import

from enum import Enum

class Type(Enum):
    EXPLICIT = 1
    IMPLICIT = 0

class Node():
    def __init__(self,value=None,label=None,type=None,score=0,level=None,id=None,pid=None,MCS=0, addr_id=None):
        self.value = value
        self.label = label
        self.type = type
        self.score = score
        self.level = level
        self.id = id if id != None else self.__hash__()
        self.addr_id = addr_id
        self.pid = pid
        self.MCS = MCS

    def __hash__(self):
        return hash(self.value + self.label)

    def assign(self,node):
        self.value = node.value
        self.type = node.type
        self.score = node.score
        self.level = node.level
        self.id = node.id
        self.MCS = node.MCS

    def print(self):
        print(self.__dict__)
