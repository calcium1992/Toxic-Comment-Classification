from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPool1D, Flatten, Dense, Dropout


class CNN(object):
    def __init__(self, classes, vocab_size, config, pretrained_embedding=None):
        self.classes, self.num_class = classes, len(classes)
        self.vocab_size = vocab_size
        self.config = config
        self.pretrained_embedding = pretrained_embedding
        self.model = self.__create_model()

    def fit(self, x_train, y_train, x_val, y_val):
        if x_val is not None:
            history = self.model.fit(
                x=x_train, y=y_train,
                epochs=self.config['epochs'],
                verbose=True,
                validation_data=(x_val, y_val),
                batch_size=self.config['batch_size'])
        else:
            history = self.model.fit(
                x=x_train, y=y_train,
                epochs=self.config['epochs'],
                verbose=True,
                batch_size=self.config['batch_size'])

    def predict(self, x_test):
        return self.model.predict(x_test) >= 0.5

    def predict_prob(self, x_test):
        return self.model.predict(x_test)

    def __create_model(self):
        model = Sequential()

        if self.pretrained_embedding:
            model.add(Embedding(
                input_dim=self.vocab_size, output_dim=self.config['embedding_dim'],
                input_length=self.config['maxlen'],
                weights=[self.pretrained_embedding],
                embeddings_initializer='uniform', trainable=True))
        else:
            model.add(Embedding(
                input_dim=self.vocab_size, output_dim=self.config['embedding_dim'],
                input_length=self.config['maxlen'],
                embeddings_initializer='uniform', trainable=True))

        model.add(Conv1D(filters=128, kernel_size=7, activation='relu', padding='same'))
        model.add(MaxPool1D())

        model.add(Conv1D(filters=256, kernel_size=5, activation='relu', padding='same'))
        model.add(MaxPool1D())

        model.add(Conv1D(filters=512, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPool1D())

        model.add(Flatten())

        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(units=self.num_class, activation=None))
        model.add(Dense(units=self.num_class, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        return model




