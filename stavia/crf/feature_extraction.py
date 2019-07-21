from __future__ import absolute_import

from ..utils.parameters import *
from ..utils.utils import ispunct

import codecs
import json

rat_dict = None

def extend_rat_features(word, features):
    global rat_dict
    if rat_dict == None:
        with codecs.open(RAT_DICT_FILE, encoding='utf8', mode='r') as f:
            rat_dict = json.load(f)

    if word.lower() in rat_dict:
        new_features = []
        for name_feature, value in rat_dict[word.lower()].items():
            new_features.append('%s=%s' % (name_feature, str(value)))
        features.extend(new_features)


def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.ispunct=%s' % ispunct(word)
    ]

    if USE_RAT is True:
        extend_rat_features(word, features)

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:word.ispunct=%s' % ispunct(word1)
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    if i > 1:
        word1 = doc[i-2][0] + doc[i-1][0]
        features.extend([
            '-2:word.lower=' + word1.lower(),
            '-2:word.istitle=%s' % word1.istitle(),
            '-2:word.isupper=%s' % word1.isupper(),
            '-2:word.isdigit=%s' % word1.isdigit(),
            '-2:word.ispunct=%s' % ispunct(word1)
        ])


    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:word.ispunct=%s' % ispunct(word1)
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features

# A function for extracting features in documents
def extract_features(doc):
	return [word2features(doc, i) for i in range(len(doc))]


