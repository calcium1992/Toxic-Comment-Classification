import re
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Preprocessor(object):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.classes = config['classes']
        self.__load_data()

    def __load_data(self):
        # Read Training Data
        train_df = pd.read_csv(self.config['input_trainset'])
        train_df[['input_text_column']].fillna('unknown', inplace=True)  # Fill blank input text.
        train_df['no_class'] = 1 - train_df[self.classes]  # If none of classes is labeled.

        # Split Validation Set
        self.x, self.y = self.__parse_data(train_df)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x, self.y,
            test_size=self.config['split_ratio'], random_state=self.config['random_seed'])
        self.y_val = np.delete(self.y_val, -1, 1)  # Remove 'no_class' in validation data.

        # Read Test Data
        test_df = pd.read_csv(self.config['input_testset'])
        test_df[['input_text_column']].fillna('unknown', inplace=True)
        self.test_ids, self.x_test = self.__parse_data(test_df, is_test=True)

    def __parse_data(self, data_df, is_test=False):




