#!/usr/bin/env python
import MeCab

def parse(text):
    '''
    tokenize a text using MeCab with mecab-ipadic-neologd
    '''
    m = MeCab.Tagger('-d /Users/otknoy/local/lib/mecab/dic/mecab-ipadic-neologd/')
    m.parse('')

    node = m.parseToNode(text)

    terms = []
    while node:
        features = node.feature.split(',')
        basic_form = features[6]

        t = basic_form
        if basic_form == '*':
            t = node.surface

        terms.append(t)

        node = node.next

    return terms[1:-1]


if __name__ == '__main__':
    import sys
    filename = sys.argv[1]

    text = open(filename).read()

    print('\n'.join(parse(text)))
