from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1d, MaxPool1D, Flatten, Dense, Dropout


class CNN(object):
    def __init__(self, classes, config):
        self.classes, self.num_class = classes, len(classes)
        self.config = config
        self.model = self.__create_model()

    def fit(self, x_train, y_train, x_val, y_val):
        history = self.model.fit(
            x=x_train, y=y_train,
            epochs=self.config['epochs'],
            verbose=True,
            validation_data=(x_val, y_val),
            batch_size=self.config['batch_size'])

    def predict(self, x_test):
        return self.model.predic(x_test)

    def predict_prob(self, x_test):
        return self.model.predic(x_test) >= 0.5

    def __create_model(self):
        model = Sequential()

        model.add(Embedding(
            import_dim=self.config['vocab_size'], output_dim=self.config['embedding_dim'],
            input_length=self.config['maxlen'],
            embeddings_initializer='uniform', trainable=True))

        model.add(Conv1d(filters=128, kernelsize=7, activation='relu', padding='same'))
        model.add(MaxPool1D())

        model.add(Conv1d(filters=256, kernelsize=5, activation='relu', padding='same'))
        model.add(MaxPool1D())

        model.add(Conv1d(filters=512, kernelsize=3, activation='relu', padding='same'))
        model.add(MaxPool1D())

        model.add(Flatten())

        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(units=self.num_class), activation=None)
        model.add(Dense(units=self.num_class), activation='sigmoid')

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.summary()
        return model




