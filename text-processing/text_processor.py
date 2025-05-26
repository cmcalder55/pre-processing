
from tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# TODO
class TextProcessor:
    def __init__(self, 
                 tokenizer: Tokenizer=Tokenizer(),
                 tfidf_vectorizer: TfidfVectorizer=TfidfVectorizer(),
                 bert_vectorizer=None,
                 bow_vectorizer=None,
                 w2v_vectorizer=None
            ):
        # tokenizer attrs
        self.tokenizer = tokenizer
        self.tokenized = None
        self.vocabulary = None
        # tf-idf attrs
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tf_idf = None
        self.dtm = None
        # other vectorizers
        self.bert_vectorizer = bert_vectorizer
        self.bow_vectorizer = bow_vectorizer
        self.w2v_vectorizer = w2v_vectorizer