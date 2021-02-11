import re
import io
import string
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Preprocessor(object):
    def __init__(self, config, logger, rebalance=False):
        self.config = config
        self.logger = logger
        self.rebalance = rebalance
        self.classes = self.config['classes']
        self.pretrained_embedding = None
        self.__load_data()

    def process(self):
        x, y = self.x, self.y
        x_train, x_val, y_train, y_val = self.x_train, self.x_val, self.y_train, self.y_val
        x_test = self.x_test

        input_convertor = self.config.get('input_convertor', None)
        if input_convertor == 'count_vectorization':
            x_train, x_val, x_test = self.__count_vectorization(x_train, x_val, x_test)
        elif input_convertor == 'tfidf_vectorization':
            x_train, x_val, x_test = self.__tfidf_vectorization(x_train, x_val, x_test)
        elif input_convertor == 'nn_vectorization':
            x_train, x_val, x_test = self.__nn_vectorization(x_train, x_val, x_test)
        else:
            self.logger.warning(f'Input Convertor {input_convertor} is not supported yet.')

        return x, y, x_train, y_train, x_val, y_val, x_test

    def __load_data(self):
        # Read Training Data
        train_df = pd.read_csv(self.config['input_trainset'])
        train_df[self.config['input_text_column']].fillna('unknown', inplace=True)  # Fill blank input text.

        # Re-balance Data
        if self.rebalance:
            positive_df = train_df[train_df[self.classes].max(axis=1) == 1]
            negative_df = train_df[train_df[self.classes].max(axis=1) == 0]
            ratio = len(negative_df) // len(positive_df)
            if ratio >= 2:
                new_train_df = negative_df.copy()
                for i in range(ratio):
                    new_train_df = new_train_df.append(positive_df.copy())

                num_pos, num_neg = len(positive_df), len(negative_df)
                new_num_pos = len(new_train_df[new_train_df[self.classes].max(axis=1) == 1])
                new_num_neg = len(new_train_df[new_train_df[self.classes].max(axis=1) == 0])
                self.logger.info(f'Rebalanced Pos-{num_pos}/Neg-{num_neg} into Pos-{new_num_pos}/Neg-{new_num_neg}.')

                train_df = new_train_df

        # Split Validation Set
        self.x, self.y = self.__parse_data(train_df)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x, self.y,
            test_size=self.config['split_ratio'], random_state=self.config['random_seed'])

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

    def __count_vectorization(self, x_train, x_val, x_test):
        vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        vectorized_x_train = vectorizer.fit_transform(x_train)  # The output is a #sample * #vocab matrix. It uses
        # sparse matrix to save: (sample i, word j): count value.
        vectorized_x_val = vectorizer.transform(x_val)
        vectorized_x_test = vectorizer.transform(x_test)
        return vectorized_x_train, vectorized_x_val, vectorized_x_test

    def __tfidf_vectorization(self, x_train, x_val, x_test):
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        vectorized_x_train = vectorizer.fit_transform(x_train)  # The output is a #sample * #vocab matrix. It uses
        # sparse matrix to save: (sample i, word j): td*idf value. tf = count word in one doc / doc length.
        # idf = log(#doc / count word in all doc).
        vectorized_x_val = vectorizer.transform(x_val)
        vectorized_x_test = vectorizer.transform(x_test)
        return vectorized_x_train, vectorized_x_val, vectorized_x_test

    def __nn_vectorization(self, x_train, x_val, x_test):
        # Build Vocab
        def add_word(word2idx_dict, idx2word_dict, word):
            if word in word2idx_dict:
                return None
            idx = len(word2idx_dict)
            word2idx_dict[word], idx2word_dict[idx] = idx, word
            return idx

        word2idx, idx2word = {}, {}
        special_tokens = ['<pad>', '<unk>']

        pretrained_embedding_file = self.config.get('pretrained_embedding_file', None)
        if pretrained_embedding_file:
            word2embedding = Preprocessor.__load_embedding_vector(pretrained_embedding_file)

            vocab = list(word2embedding.keys()) + special_tokens
            for token in special_tokens:
                word2embedding[token] = np.random.uniform(low=-1, high=1, size=self.config['embedding_dim'])

            self.vocab_size = len(vocab)
            self.pretrained_embedding = np.zeros(shape=(self.vocab_size, self.config['embedding_dim']))
            for word in vocab:
                idx = add_word(word2idx, idx2word, word)
                if idx is not None:
                    self.pretrained_embedding[idx] = word2embedding[word]
        else:
            for token in special_tokens:
                add_word(word2idx, idx2word, token)

            for sentence in x_train:
                for word in sentence:
                    add_word(word2idx, idx2word, word)
            self.vocab_size = len(word2idx)

        # Translate Sentences
        def vectorize(dataset, word2idx_dict):
            dataset_ids = []
            for sentence in dataset:
                ids = [word2idx_dict.get(word, word2idx['<unk>']) for word in sentence]
                dataset_ids.append(ids)
            return np.array(dataset_ids)

        x_train_ids = vectorize(x_train, word2idx)
        x_val_ids = vectorize(x_val, word2idx)
        x_test_ids = vectorize(x_test, word2idx)

        x_train_ids = keras.preprocessing.sequence.pad_sequences(
            x_train_ids, maxlen=self.config['maxlen'],
            padding='post', value=word2idx['<pad>'])
        x_val_ids = keras.preprocessing.sequence.pad_sequences(
            x_val_ids, maxlen=self.config['maxlen'],
            padding='post', value=word2idx['<pad>'])
        x_test_ids = keras.preprocessing.sequence.pad_sequences(
            x_test_ids, maxlen=self.config['maxlen'],
            padding='post', value=word2idx['<pad>'])

        return x_train_ids, x_val_ids, x_test_ids

    @staticmethod
    def __load_embedding_vector(embedding_file):
        file = io.open(embedding_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        data = {}
        for line in file:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array(list(map(float, tokens[1:])))
        return data





