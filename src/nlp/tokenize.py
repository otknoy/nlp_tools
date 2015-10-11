#!/usr/bin/env python
import MeCab

def tokenize(text):
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
    def load(filename):
        return open(filename).read().split('\n')

    docs = [u'山下さんは山下くんと東京特許許可局へ行った。',
            u'山下さんは山下くんと北海道へ行った。',
            u'山下さんは下山くんと New York へ行った。',
            u'山上さんは山下くんと東京特許許可局へ行った。',]  

    for d in docs:
        terms = tokenize(d)
        print(terms)
                         

    # # kmeans 
    # features, vectorizer = tfidf(docs)
    # featuers = reduction(features, dim=1000)
    # # print(features.toarray())

    # labels = kmeans(features, k=1000)

    # output = ['%02d,%s' % (l, d) for d, l in sorted(zip(docs, labels), key=lambda x:x[1])]
    # open('test.csv', 'w').write('\n'.join(output))


    # docs = [tokenize(d) for d in docs]
    # model, topics = lda(docs, k=1000)
    # for k in topics:
    #     print(' + '.join(["%s*%s" % (v, term) for v, term in k]))
    
    
