import numpy as np
import util.misc

class DataHandlerException(Exception):
    pass

class BaseDataHandler(object):

    DATA_ALL = 1
    DATA_TRAIN = 2
    DATA_VALIDATION = 4
    DATA_TEST = 3

    def __init__(self, source):
        self.source = source

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
    def to_word_level_vectors(texts_list, wv_container, words_per_text=None):
        """
        Receives a list of texts. For each text, it converts the text into a list of word vectors
        given by a vector container (Glove, WordToVec) for direct use as input

        If words_per_text is specified, each text representation can have as many
        as words_per_text words. Hence texts will be cut or zero-padded.
        """
        from util.language import tokenize_text

        tokenized_texts = util.misc.parallel_run(tokenize_text, texts_list)
        text_wvs_indices = [wv_container.get_indices(text) for text in tokenized_texts]
        del tokenized_texts

        text_wvs = [wv_container[text_indices] for text_indices in text_wvs_indices]
        del text_wvs_indices

        if words_per_text is not None:
            text_wvs = BaseDataHandler.__pad_sequence_word_vectors(text_wvs, words_per_text)
        return text_wvs

    @staticmethod
    def __pad_sequence_word_vectors(text_wvs, maxlen=None):
        """
        Given a list of lists of word vectors (this is, wvs for texts), it zero-pads
        or reduces the number of words up to maxlen if specified. Otherwise, it pads
        everything up to the maximum text size
        """
        lengths = [len(s) for s in text_wvs]

        nb_samples = len(text_wvs)
        if maxlen is None:
            maxlen = np.max(lengths)

        wv_dim = text_wvs[0].shape[1]
        x = np.zeros((nb_samples, maxlen, wv_dim)).astype('float32')
        for idx, s in enumerate(text_wvs):
            x[idx, :lengths[idx]] = s[:maxlen]
        return x

    @staticmethod
    def to_char_level_idx(texts_list, char_container, chars_per_word=None, words_per_document=None, prepend=False):
        """
        Receives a list of texts. For each text, it converts the text into a list of indices of a characters
        for later use in the embedding of a neural network.
        Texts are padded (or reduced) up to chars_per_word

        char_container is assumed to be a method that converts characters to indices using a method
        called get_indices()
        """
        from util.language import tokenize_text
        texts_list = util.misc.parallel_run(tokenize_text, texts_list)

        if words_per_document is not None:
            text_with_indices = [BaseDataHandler.__normalize(char_container.get_indices(txt), chars_per_word, prepend) for txt in texts_list]
            text_with_indices = BaseDataHandler.__normalize(text_with_indices, size=words_per_document, filler=[0] * chars_per_word)
        else:
            text_with_indices = char_container.get_indices(texts_list)
        return text_with_indices

    @staticmethod
    def to_word_level_idx(texts_list, wv_container, words_per_document=None, prepend=False):
        """
        Receives a list of texts. For each text, it converts the text into indices of a word
        vector container (Glove, WordToVec) for later use in the embedding of a neural network.
        Texts are padded (or reduced) up to words_per_document
        """
        from util.language import tokenize_text
        texts_list = util.misc.parallel_run(tokenize_text, texts_list)

        if words_per_document is not None:
            text_with_indices = BaseDataHandler.__normalize(wv_container.get_indices(texts_list), words_per_document, prepend)
        else:
            text_with_indices = wv_container.get_indices(texts_list)
        return text_with_indices

    @staticmethod
    def to_sentence_level_idx(texts_list, sentences_per_paragraph, words_per_sentence, wv_container, prepend=False):
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
        texts_list = util.misc.parallel_run(parse_paragraph, texts_list)
        text_with_normalized_sentences = [BaseDataHandler.__normalize(review, size=words_per_sentence, prepend=prepend)
                                          for review in wv_container.get_indices(texts_list)]
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
