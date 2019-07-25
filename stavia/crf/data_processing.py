from __future__ import absolute_import

from ..utils.parameters import *

import codecs


def load_data(filepath):
	docs = []
	element_size = 0
	X = []
	Y = []
	with codecs.open(filepath, encoding='utf8', mode='r') as f:
		data_string_list = list(f)
		for data_string in data_string_list:
			words = data_string.strip().split()
			if len(words) is 0:
				docs.append((X,Y))
				X = []
				Y = []
			else:
				if element_size is 0:
					element_size = len(words)
				elif element_size is not len(words):
					continue
				X.append(words[:-1])
				Y.append(words[-1])

		if len(X) > 0:
			docs.append((X,Y))

	return docs


def tokenize(inp):
	inp = inp.strip()
	words = []
	word = ''
	for char in inp:
		if char == ' ':
			if len(word) != 0:
				words.append(word)
				word = ''
		elif char in PUNCTUATIONS:
			if len(word) != 0:
				words.append(word)
				word = ''
			words.append(char)
		else:
			word += char
	if len(word) != 0:
		words.append(word)

	return words

def wrap_postag(words):
	ret = []
	for word in words:
		ret.append([word,'O'])
	return ret