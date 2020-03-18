"""
Homework #3

Implement an n-gram language model that can separate true sentences from the artificially obtained sentences.

"""

import numpy as np
import pandas as pd
ALPHA = 0.01

class LanguageModel(object):
    """
        n-gramm model
    """

    def __init__(self, ngram_size=2):

        if ngram_size < 2:
            raise Exception

        self.ngram_size = ngram_size

        #keys of dictionary are all words that was read by model, values are their ids (tokens)
        self.dictionary = {}
        self.number_of_words = 0

        #counters has n-grams and {n-1)-grams as keys and number of their occurances in train set as values
        self.counter = {}
        self.context_counter = {}

        self.smoothing = 'laplace'

    def fit(self, sentences):
        """
            Model training on sentence-splitted text
            :param sentences: the list of sentences
        """
        for sentence in sentences:
            self.fit_sentence(self.tokenize_sentence(sentence))

    def tokenize_sentence(self, sentence):
        """
            Getting the list of tokens by the sentence
            :return: tokenized sentence
        """


        sentence = sentence.split(" ")
        result = []

        for word in sentence:
            token = self.dictionary.get(word)

            #if word is not in dictionary, then we should add it and set a token to it
            if token is None:
                token = self.number_of_words
                self.dictionary.setdefault(word, token)
                self.number_of_words = self.number_of_words + 1

            result.append(token)

        return result

    def fit_sentence(self, sentence):
        """
            Fitting a sentence to a model
        """

        l = len(sentence)

        #we should count ever n-gram in the sentence
        for i in range(l - self.ngram_size + 1):
            ngram = tuple(sentence[i: i + self.ngram_size])
            val = self.counter.get(ngram, 0) + 1
            self.counter.update([(ngram, val)])

        #... and do the same with {n-1}-grams, which are contexts for n-grams
        #TODO: Task 1


    def ngram_prob(self, ngram):
        """
            Counting the probability of n-gram by knowing the context
        """
        if(self.smoothing == 'laplace'):

            #context for a n-gram is this n-gram without last word (token)
            context = ngram[:-1]

            #amount of unique {n-1}-grams
            V = len(self.context_counter.keys())

            #amount of occurances of given n-gram and its context in train set
            ngram_count = self.counter.get(ngram, 0)
            context_count = self.context_counter.get(context, 0)

            #TODO: TASK 3

    def sentence_logprob(self, sentence):
        """
            Counting the log of probability of the given sentence as sum of log probabilities of its n-grams
        """

        sentence = self.tokenize_sentence(sentence)
        l = len(sentence)
        logprob = 0

        #TODO: Task 2

        return logprob

    def log_prob(self, sentences):
        return [self.sentence_logprob(sentence) for sentence in sentences]

df_train = pd.read_csv("train.tsv", sep='\t')
df_test = pd.read_csv("task.tsv", sep='\t')

print(df_train.head(2))

print("Read ", df_train.shape, df_test.shape)

basic_lm = LanguageModel()

sentences_train = df_train["text"].tolist()
basic_lm.fit(sentences=sentences_train)

print("Trained")

test1, test2 = df_test["text1"], df_test["text2"]

logprob1, logprob2 = np.array(basic_lm.log_prob(test1)), np.array(basic_lm.log_prob(test2))

res = pd.DataFrame()
res["id"] = df_test["id"]
res["which"] = 0
res.loc[logprob2 >= logprob1, ["which"]] = 1

res.to_csv("submission.csv", sep=",", index=None, columns=["id", "which"])
