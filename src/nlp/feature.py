#!/usr/bin/env python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

def tf(doc):
    docs = [' '.join(d) for d in docs]

    vectorizer = CountVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    features = vectorizer.fit_transform(doc)
    terms = vectorizer.get_feature_names()
    return features, terms

def tfidf(docs):
    docs = [' '.join(d) for d in docs]

    vectorizer = TfidfVectorizer(min_df=1, max_df=50, token_pattern=u'(?u)\\b\\w+\\b')
    features = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names()

    return features, terms

def reduction(x, dim=10):
    '''
    dimensionality reduction using LSA
    '''
    lsa = TruncatedSVD()
    x = lsa.fit_transform(x)
    x = Normalizer(copy=False).fit_transform(x)
    return x


if __name__ == '__main__':
    docs = [['山下', 'さん', 'は', '山下', 'くん', 'と', '東京特許許可局', 'へ', '行く', 'た', '。'],
            ['山下', 'さん', 'は', '山下', 'くん', 'と', '北海道', 'へ', '行く', 'た', '。'],
            ['山下', 'さん', 'は', '下山', 'くん', 'と', 'New York', 'へ', '行く', 'た', '。'],
            ['山上', 'さん', 'は', '山下', 'くん', 'と', '東京特許許可局', 'へ', '行く', 'た', '。'],]

    features, terms = tfidf(docs)
    print(terms)
    print(features.toarray())

    features, terms = tfidf(docs)
    print(terms)
    print(features.toarray())

    features = reduction(features, dim=2)
    print(features)
