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


from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(docs):
    vectorizer = TfidfVectorizer(analyzer=tokenize, min_df=1, max_df=50)

    features = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names()

    return features, vectorizer


from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

def reduction(x, dim=10):
    '''
    dimensionality reduction using LSA
    '''
    lsa = TruncatedSVD()
    x = lsa.fit_transform(x)
    x = Normalizer(copy=False).fit_transform(x)
    return x


from sklearn.cluster import KMeans

def kmeans(features, k=10):
    km = KMeans(n_clusters=k, init='k-means++', n_init=1, verbose=True)
    km.fit(features)
    return km.labels_


if __name__ == '__main__':
    def load(filename):
        return open(filename).read().split('\n')

    import sys
    filename = sys.argv[1]
    docs = load(filename)
    # docs = [u'山下さんは山下くんと東京特許許可局へ行った。',
    #          u'山下さんは山下くんと北海道へ行った。',
    #          u'山下さんは下山くんと New York へ行った。',
    #          u'山上さんは山下くんと東京特許許可局へ行った。',]  

    features, vectorizer = tfidf(docs)
    featuers = reduction(features, dim=1000)
    # print(features.toarray())

    labels = kmeans(features, k=1000)

    output = ['%02d,%s' % (l, d) for d, l in sorted(zip(docs, labels), key=lambda x:x[1])]
    open('test.csv', 'w').write('\n'.join(output))

    
