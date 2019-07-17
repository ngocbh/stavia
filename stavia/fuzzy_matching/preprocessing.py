from __future__ import absolute_import

import re

def get_duplicate(addr, n):
    list_ngram = generate_ngrams(addr,n)
    dup = {}
    temp = []
    for i in list_ngram:
        if i in temp and i in dup.keys():
            dup.update({i : dup.get(i)+1})
        elif i in temp:
            dup.update({i : 1})
        else:
            temp.append(i)
    return dup

def generate_ngrams(s, n):
    s = s.lower()
    tokens = [token for token in s.split(" ") if token != ""]
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]


def preprocess_address(addr):
    addr = addr.lower()
    addr = addr.replace(". "," ").replace("."," ")
    addr = addr.replace(", "," ").replace(","," ")
    addr = ' '.join(addr.replace(",",", ").replace("-"," - ").split())
    addr = addr.replace("Thành phố","")
    addr = addr.replace("thành phố","")
    addr = addr.replace("Thành Phố","")
    addr = addr.replace("Thanh pho ","")
    addr = addr.replace("thanh pho ","")
    addr = addr.replace("Thanh Pho ","")
    isShort = re.search(r'(\.\w+)', addr)
    if isShort:
        addr = addr.replace(".", ". ")

    # remove duplicate n-gram
    max_gram = int(len(addr.split(" "))/2)
    for i in range(2, max_gram+1):
        duplicate = get_duplicate(addr,i)
        if len(duplicate.keys()) != 0:
            for dup_word in duplicate.keys():
                addr = addr.replace(dup_word,"",duplicate.get(dup_word))

    return addr.replace("  "," ",10)
