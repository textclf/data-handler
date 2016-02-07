'''
base.py -- ABC for data handler.
'''


import abc

import numpy as np

from .util.parallel import parallel_run

class DataHandlerException(Exception):
    pass

class BaseDataHandler(object):

    __metaclass__  = abc.ABCMeta

    DATA_ALL = 1
    DATA_TRAIN = 2
    DATA_VALIDATION = 4
    DATA_TEST = 3

    def __init__(self, source):
        self.source = source

    @abc.abstractmethod
    def get_data(self, type=DATA_ALL):
        """
        Process the data from its source and returns two lists: texts and labels, ready for a classifier to be used
        """
        raise NotImplementedError()

    @staticmethod
    def shuffle_data(train_values, labels):
        combined_lists = zip(train_values, labels)
        np.random.shuffle(combined_lists)
        return zip(*combined_lists)

    @staticmethod
    def word_level_ix(texts_list, words_per_document, wv_container, prepend=False, needs_tokenizing=False):
        """
        Receives a list of texts. For each text, it converts the text into indices of a word 
        vector container (Glove, WordToVec) for later use in the embedding of a neural network.

        Sentences are padded (or reduced) up to words_per_sentence elements.
        Texts ("paragraphs") are padded (or reduced) up to sentences_per_paragraph
        If prepend = True, padding is added at the beginning

        Ex: [[This might be cumbersome. Hopefully not.], [Another text]]
               to
            [  [[5, 24, 3, 223], [123, 25, 0, 0]]. [[34, 25, 0, 0], [0, 0, 0, 0]  ]
            using sentences_per_paragraph = 4, words_per_sentence = 4
        """

        

        if needs_tokenizing:
            from util.language import tokenize_text
            texts_list = parallel_run(tokenize_text, texts_list)

        text_with_normalized_documents = BaseDataHandler.__normalize(wv_container.get_indices(texts_list), words_per_document, prepend)
        return text_with_normalized_documents


    @staticmethod
    def sentence_level_ix(texts_list, sentences_per_paragraph, words_per_sentence, wv_container, prepend=False):
        """
        Receives a list of texts. For each text, it converts the text into sentences and converts the words into
        indices of a word vector container (Glove, WordToVec) for later use in the embedding of a neural network.

        Sentences are padded (or reduced) up to words_per_sentence elements.
        Texts ("paragraphs") are padded (or reduced) up to sentences_per_paragraph
        If prepend = True, padding is added at the beginning

        Ex: [[This might be cumbersome. Hopefully not.], [Another text]]
               to
            [  [[5, 24, 3, 223], [123, 25, 0, 0]]. [[34, 25, 0, 0], [0, 0, 0, 0]  ]
            using sentences_per_paragraph = 4, words_per_sentence = 4
        """

        from util.language import parse_paragraph

        text_sentences = parallel_run(parse_paragraph, texts_list)

        text_with_normalized_sentences = [BaseDataHandler.__normalize(review, size=words_per_sentence, prepend=prepend)
                                          for review in wv_container.get_indices(text_sentences)]
        text_padded_paragraphs = BaseDataHandler.__normalize(text_with_normalized_sentences,
                                                      size=sentences_per_paragraph, filler=[0] * words_per_sentence)

        return text_padded_paragraphs

    @staticmethod
    def __normalize(sq, size=30, filler=0, prepend=False):
        """
        Take a list of lists and ensure that they are all of length `sz`

        Args:
        -----
        e: a non-generator iterable of lists
        sz: integer, the size that each sublist should be normalized to
        filler: obj -- what should be added to fill out the size?
        prepend: should `filler` be added to the front or the back of the list?
        """
        if not prepend:
            def _normalize(e, sz):
                return e[:sz] if len(e) >= sz else e + [filler] * (sz - len(e))
        else:
            def _normalize(e, sz):
                return e[-sz:] if len(e) >= sz else [filler] * (sz - len(e)) + e
        return [_normalize(e, size) for e in sq]
