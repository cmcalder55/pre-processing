import pandas as pd
import numpy as np
import nltk

def analyze_dtm(dtm, words, sents):
    """
    * takes an array $dtm$ and $words$ as an input, where $dtm$ is the array you get in Q3 with a shape $(m times n)$,
    and $words$ contains an array of words corresponding to the columns of $dtm$.
    * calculates the sentence frequency for each word, say $j$, e.g. how many sentences contain word $j$. Save the result
    to array $df$ ($df$ has shape of $(n,)$ or $(1, n)$).
    * normalizes the word count per sentence: divides word count, i.e., $dtm_{i,j}$, by the total number of words in
    sentence $i$. Save the result as an array named $tf$ ($tf$ has shape of $(m,n)$).
    * for each $dtm_{i,j}$, calculates $tf/_idf_{i,j} = frac{tf_{i, j}}{df_j}$, i.e., divide each normalized word count by 
    the sentence frequency of the word. The reason is, if a word appears in most sentences, it does not have the discriminative 
    power and often is called a `stop` word. The inverse of $df$ can downgrade the weight of such words. $tf/_idf$ has shape of $(m,n)$
    * prints out the following:

        - the total number of words in the document represented by $dtm$
        - the most frequent top 10 words in this document, compare with the results from Q2, and briefly explain the difference
        - words with the top 10 largest $df$ values (show words and their $df$ values)
        - the longest sentence (i.e., the one with the most words)
        - top-10 words with the largest $tf/_idf$ values in the longest sentence (show words and values)
    * returns the $tf/_idf$ array.
    """

    df = np.count_nonzero(dtm, axis=0)

    total = dtm.sum(axis=1)[:, np.newaxis]
    tf = dtm/total

    tfidf = tf/df[np.newaxis,:]

    n_top = 10

    words_freq = dtm.sum(axis=0)
    words_most = words_freq.argsort()[::-1][:n_top]
    top_words = list(zip(words[words_most], words_freq[words_most]))

    hi_df = df.argsort()[::-1][:n_top]
    top_df = list(zip(words[hi_df], df[hi_df]))

    longest = dtm.sum(axis=1).argmax()

    longest_tfidf = tfidf[longest].argsort()[::-1][:n_top]
    top_tfidf = list(zip(words[longest_tfidf], tfidf[longest][longest_tfidf]))

    print(f'The total number of words:\n{dtm.sum()}\n')
    print(f'The top 10 frequent words:\n{top_words}\n')
    print(f'The top 10 words with highest df values:\n{top_df}\n')
    print(f'The longest sentence:\n{sents[longest]}\n')
    print(f'The top 10 words with highest tf-idf values in the longest sentence:\n{top_tfidf}')

    return tfidf

def evaluate_performance(prob, truth, th):
    """Compares the prediction with the ground truth labels to calculate
    the confusion matrix as [[TN, FN],[FP,TP]], where:
    * True Positives (TP): the number of correct positive predictions
    * False Positives (FP): the number of postive predictives which actually are negatives
    * True Negatives (TN): the number of correct negative predictions
    * False Negatives (FN): the number of negative predictives which actually are positives

    :return precision: TP/(TP+FP)
    :return recall: TP/(TP+FN)
    """
    
    conf = [[0, 0], [0, 0]]
    classifiers = list(zip(prob, truth))

    for p,t in classifiers:
        guess = 0
        if p > th:
            guess = 1
        if guess == t:
                if guess == 0:
                    conf[0][0] += 1
                else:
                    conf[1][1] += 1
        else:
            if guess == 0:
                conf[0][1] += 1
            else:
                conf[1][0] += 1

    [[TN,FN],[FP,TP]] = conf
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)

    performance = {
        "R0": prec,
        "R1": rec
    }
    return performance

def bigram_precision_recall(gen_tokens, ref_tokens):
    result = pd.DataFrame(columns = ['overlapping','precision','recall'])

    gen_bigrams = [list(nltk.bigrams(tokens)) for tokens in gen_tokens]
    ref_bigrams = [list(nltk.bigrams(tokens)) for tokens in ref_tokens]

    bigrams = list(zip(gen_bigrams, ref_bigrams))

    overlapping = []
    precision = []
    recall = []
    for gen, ref in bigrams:
        overlap = [tup1 for tup1 in gen for tup2 in ref if tup1 == tup2]
        overlapping.append(list(set(overlap)))

        prec = 0
        if gen:
            prec = len(overlap)/len(gen)
        precision.append(prec)

        rec = 0
        if ref:
            rec = len(overlap)/len(ref)
        recall.append(rec)

    result['overlapping'] = overlapping
    result['precision'] = precision
    result['recall'] = recall

    return result
