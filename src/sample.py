#!/usr/bin/env python
if __name__ == '__main__':
    import sys
    filename = sys.argv[1]
    texts = open(filename).read().split('\n')

    from nlp.tokenize import tokenize
    from nlp.feature import tfidf
    
    docs = [tokenize(t) for t in texts]
    features, terms = tfidf(docs)
    print(terms)
    print(features)
    
