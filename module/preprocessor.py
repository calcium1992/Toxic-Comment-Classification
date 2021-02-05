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

    def process(self):
        x, y = self.x, self.y
        x_train, x_val, y_train, y_val = self.x_train, self.x_val, self.y_train, self.y_val
        x_test = self.x_test

        input_convertor = self.config.get('input_convertor', None)
        if input_convertor == 'count_vectorization':
            x_train, x_val = self.__count_vectorization(x_train, x_val)
            x, x_test = self.__count_vectorization(x, x_test)
        elif input_convertor == 'tfidf_vectorization':
            x_train, x_val = self.__tfidf_vectorization(x_train, x_val)
            x, x_test = self.__tfidf_vectorization(x, x_test)

        return x, y, x_train, y_train, x_val, y_val, x_test

    def __load_data(self):
        # Read Training Data
        train_df = pd.read_csv(self.config['input_trainset'])
        train_df[self.config['input_text_column']].fillna('unknown', inplace=True)  # Fill blank input text.
        train_df['no_class'] = 1 - train_df[self.classes].max(axis=1)  # If none of classes is labeled.

        # Split Validation Set
        self.x, self.y = self.__parse_data(train_df)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x, self.y,
            test_size=self.config['split_ratio'], random_state=self.config['random_seed'])
        self.y_val = np.delete(self.y_val, -1, 1)  # Remove 'no_class' in validation data.

        # Read Test Data
        test_df = pd.read_csv(self.config['input_testset'])
        test_df[self.config['input_text_column']].fillna('unknown', inplace=True)
        self.test_ids, self.x_test = self.__parse_data(test_df, is_test=True)

    def __parse_data(self, data_df, is_test=False):
        x = data_df[self.config['input_text_column']].apply(Preprocessor.__clean_text).values
        if is_test:
            test_ids = data_df['id'].values
            return test_ids, x
        else:
            y = data_df.drop([self.config['input_id_column'], self.config['input_text_column']], 1).values
            return x, y

    @staticmethod
    def __clean_text(text):
        text = text.strip().lower().replace('\n', '')  # Trim, lower and remove newline character.
        words = re.split(r'\W+', text)  # Split for list of words.
        punc_filter = str.maketrans('', '', string.punctuation)
        words = [w.translate(punc_filter) for w in words if len(w.translate(punc_filter))]  # Remove punctuations.
        return words

    @staticmethod
    def __count_vectorization(x_train, x_test):
        vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        vectorized_x_train = vectorizer.fit_transform(x_train)  # The output is a #sample * #vocab matrix. It uses
        # sparse matrix to save: (sample i, word j): count value.
        vectorized_x_test = vectorizer.transform(x_test)
        return vectorized_x_train, vectorized_x_test

    @staticmethod
    def __tfidf_vectorization(x_train, x_test):
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        vectorized_x_train = vectorizer.fit_transform(x_train)  # The output is a #sample * #vocab matrix. It uses
        # sparse matrix to save: (sample i, word j): td*idf value. tf = count word in one doc / doc length.
        # idf = log(#doc / count word in all doc).
        vectorized_x_test = vectorizer.transform(x_test)
        return vectorized_x_train, vectorized_x_test
