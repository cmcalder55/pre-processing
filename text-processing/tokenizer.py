
class Tokenizer():
    def __init__(self, mode="basic"):
        self.mode = mode
        self.vocab = {}
        self.tokenized = None

    def tokenize_basic(self, docs, reset_vocabulary=False):
        """Tokenizes a sentence to get tokens and vocab freq dict
        * accepts a sentence (i.e., `text` parameter) as an input
        * splits the sentence into a list of tokens by **space**
        * removes the **leading/trailing punctuations or spaces** of each token, if any
        * only keeps tokens with 2 or more characters, i.e. `len(token)>1`
        * converts all tokens into lower case
        * find the count of each unique token and save the counts as dictionary, i.e., `{world: 1, a: 1, ...}`
        :param str text: sentence (corpus)
        """

        punct = punctuation + '\u201c\u201d\u2018\u2019'
        # Remove any leading or trailing punctuation and spaces/tabs/new lines from each
        # If a word is longer than 2 letters, covert to lower case and keep
        tokens = [
            [t.lower().strip(punct) for t in doc.split() if len(t) > 1]
            for doc in docs
        ]
        # Store word count in vocab dict
        vocab = {}
        for doc in tokens:
            for t in doc:
                if t not in vocab:
                    vocab[t] = 1.0
                else:
                    vocab[t] += 1.0
            if reset_vocabulary:
                self.vocab = vocab
            else:
                self.vocab.update(vocab)
        
        self.vocab = dict(sorted(self.vocab.items(), key=lambda v: v[1], reverse=True))
        
        return tokens

    def tokenize_doc(self, doc, nlp, lemmatized=True, remove_stopword=True, remove_punct=True):
        clean_tokens = []
        # load current doc into spacy nlp model and split sentences by newline chars
        sentences = doc.split("\\n")
        for sentence in sentences:
            doc = nlp(sentence)

            # clean either lemmatized unigrams or unmodified doc tokens
            if lemmatized:
                clean_tokens += [token.lemma_.lower() for token in doc            # using spacy nlp params, skip token if:
                                if (not remove_stopword or not token.is_stop)     # it is a stopword and remove_stopwords = True
                                and (not remove_punct or not token.is_punct)      # it is punctuation and remove_punct = True
                                and not token.lemma_.isspace()]                   # it is whitespace
            else:
                clean_tokens += [token.text.lower() for token in doc
                                if (not remove_stopword or not token.is_stop)
                                and (not remove_punct or not token.is_punct)
                                and not token.text.isspace()]

        return clean_tokens

    def tokenize_spacy(self, docs, lemmatized=True, remove_stopword=True, remove_punct=True):
        """Tokenize documents using methods from the SpaCy library.
        (ref: https://spacy.io/api/token#attributes)
        
        Splits each input document into unigrams and also clean up tokens as follows:
        - if `lemmatized` is turned on, lemmatize all unigrams.
        - if `remove_stopword` is set to True, remove all stop words.
        - if `remove_punct` is set to True, remove all punctuation tokens.
        - remove all empty tokens and lowercase all the tokens.
        
        Parameters:
            docs (List[str]): a list of documents.
            lemmatized (bool): optional parameter to indicate if tokens are lemmatized. Defaults to True.
            remove_stopword(bool): optional parameter to remove stop words. Defaults to True.
            remove_punct(bool): optional parameter to remove punctuation. Defaults to True.
  
        Returns:
            out (list): tokens obtained for each document after all the processing.
        """

        # load in spacy NLP model and disable unused pipelines to reduce processing time/memory space
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.add_pipe("sentencizer")
        # tokenize each doc in the corpus using specified params for lemmatization and removal conditions
        tokens = [self.tokenize_doc(doc, nlp, lemmatized, remove_stopword, remove_punct) for doc in docs]
        return tokens

    def tokenize(self, docs, **kwargs):
        
        tokens = []
        if self.mode == "basic":
            tokens = self.tokenize_basic(docs, **kwargs)
        elif self.mode == "spacy":
            tokens = self.tokenize_spacy(docs, **kwargs)
        self.tokenized = tokens
        return tokens