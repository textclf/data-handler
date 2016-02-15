from base_handler import BaseDataHandler, DataHandlerException

import glob
import os
import cPickle as pickle

class YelpDataHandler(BaseDataHandler):
    """
    Works with Yelp reviews data from the Yelp Dataset Challenge, previously saved in
    balanced classes as inh

    https://github.com/alfredolainez/yelp-dataset
    """
    def __init__(self):
        pass

    def get_data(self, train_file, dev_file, test_file):
        """
        Gets training and test sets, merging original train and dev sets for training.

        Returns (training_reviews, train_labels, test_reviews, test_labels)
        """

        (train_reviews, train_labels) = pickle.load(open(train_file, 'rb'))
        (dev_reviews, dev_labels) = pickle.load(open(dev_file, 'rb'))
        (test_reviews, test_labels) = pickle.load(open(test_file, 'rb'))

        train_reviews = list(train_reviews)
        train_labels = list(train_labels)
        dev_reviews = list(dev_reviews)
        dev_labels = list(dev_labels)
        test_reviews = list(test_reviews)
        test_labels = list(test_labels)

        train_reviews.extend(dev_reviews)
        train_labels.extend(dev_labels)

        return (train_reviews, train_labels, test_reviews, test_labels)