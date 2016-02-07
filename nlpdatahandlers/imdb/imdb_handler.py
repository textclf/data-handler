from ..base import BaseDataHandler, DataHandlerException

import glob
import os

class ImdbDataHandler(BaseDataHandler):
    """
    Works with the original Large Movie Review Dataset - IMDB data as downloaded from
    http://ai.stanford.edu/~amaas/data/sentiment/

    source defines the folder where the data is downloaded

    Args:
    -----
        source: the path to the root aclImdb/ folder for the downloaded data

    Examples:
    ---------

        >>> imdb = ImdbHandler('./aclImdb')
        >>> train_data, train_labels = imdb.get_data()
    """

    def get_data(self, type=BaseDataHandler.DATA_TRAIN, shuffle=True):
        """
        Process the data from its source and returns two lists: texts and labels, ready for a classifier to be used

        Data is not shuffled
        """
        if type not in (BaseDataHandler.DATA_TRAIN, BaseDataHandler.DATA_TEST):
            raise DataHandlerException("Only train and test data supported for ImdbDataHandler")
        else:
            which_data = 'train' if type == BaseDataHandler.DATA_TRAIN else 'test'

        positive_examples = glob.glob(os.path.join(self.source, which_data, 'pos', '*.txt'))
        negative_examples = glob.glob(os.path.join(self.source, which_data, 'neg', '*.txt'))

        data = []
        labels = []
        for i, f in enumerate(positive_examples):
            data.append((open(f, 'rb').read().lower()).replace('<br /><br />', '\n'))
            labels.append(1)
        for i, f in enumerate(negative_examples):
            data.append((open(f, 'rb').read().lower()).replace('<br /><br />', '\n'))
            labels.append(0)

        if shuffle:
            return self.shuffle_data(data, labels)
        return (data, labels)

