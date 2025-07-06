import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizer import Tokenizer
from scipy.stats import wilcoxon

def get_dtm(sents):
    """
    - accepts a list of sentences, i.e., `sents`, as an input
    - call `tokenize` function you defined in Q1 to get the count dictionary for each sentence, and combine them into a list
    - call `generate_vocab` function in Q2 to generate the large vocabulary for all sentences, and get all the words, i.e., keys
    - creates a numpy array, say `dtm` with a shape (# of docs x # of unique words), and set the initial values to 0.
    - fills cell `dtm[i,j]` with the count of the `j`th word in the `i`th sentence. HINT: you can loop through the list of 
    vocabulary from step 2, and check each word's index in the large vocabulary from step 3, so that you can put the 
    corresponding value into the correct cell.
    - returns `dtm` and `unique_words`
    """

    tokenizer = Tokenizer("spacy")
    tokenizer.tokenize(sents, lemmatized=True, remove_stopword=True, remove_punct=True)
    
    all_docs = tokenizer.tokenized
    all_words = tokenizer.vocab

    m,n = len(all_docs), len(all_words)
    dtm = np.zeros((m,n))

    for doc in range(m):
        for i, word in enumerate(all_words.keys()):
            if word in all_docs[doc]:
                dtm[doc,i] = all_docs[doc][word]

    return dtm, all_words, tokenizer

def gen_tfidf(texts, min_df=1.0, max_df=1.0, ngram_range=(1, 1)):
    tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    features = tfidf.fit_transform(texts)
    return pd.DataFrame(features.todense(),
                        columns=tfidf.get_feature_names_out())

def compute_sentiment(target, pos, neg):
    sentiment = 0
    p = sum(1 for word in target if word in pos)
    n = sum(1 for word in target if word in neg)
    if p + n != 0:
        sentiment = (p - n) / (p + n)
    return sentiment

def sentiment(gen_tokens, ref_tokens, pos, neg):
    print(gen_tokens)
    tokens = lambda token_list: [compute_sentiment(sublist, pos, neg) for sublist in token_list]

    result = pd.DataFrame({'gen_sentiment': tokens(gen_tokens), 
                           'ref_sentiment': tokens(ref_tokens)})

    avg = (result['gen_sentiment'] - result['ref_sentiment']).mean()
    res = wilcoxon(result['gen_sentiment'] - result['ref_sentiment'], alternative='greater')

    print(f"Average Sentiment: {avg}\n")
    print(f"Stat: {res.statistic}\nP-Value: {res.pvalue}\n")
    return result
