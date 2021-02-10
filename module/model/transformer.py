import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embedding_dim, pretrained_embedding=None):
        super().__init__()
        self.maxlen, self.vocab_size, self.embedding_dim = maxlen, vocab_size, embedding_dim
        # Token Embedding
        if pretrained_embedding is not None:
            self.token_emb = layers.Embedding(
                input_dim=self.vocab_size, output_dim=self.embedding_dim,
                weights=[pretrained_embedding],
                input_length=self.maxlen, trainable=False)
        else:
            self.token_emb = layers.Embedding(
                input_dim=self.vocab_size, output_dim=self.embedding_dim,
                input_length=self.maxlen, trainable=True)
        # Position Embedding
        self.pos_emb = layers.Embedding(input_dim=embedding_dim, output_dim=embedding_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        token = self.token_emb(x)
        return token + positions


class SelfAttention(layers.Layer):
    def __init__(self, embedding_dim, num_heads=8):
        super().__init__()
        if embedding_dim % num_heads != 0:
            # Note: The input and output are both seq_len * embedding_dim. For example, if num_heads =2, then
            # the first attention head will generate the first embedding_dim / 2 elements in the output.
            raise ValueError(f'Embedding dimension {embedding_dim} should be divisible by number of heads {num_heads}.')
        self.embedding_dim, self.num_heads = embedding_dim, num_heads
        self.project_dim = int(self.embedding_dim / self.num_heads)
        self.query_dense = layers.Dense(units=embedding_dim)
        self.key_dense = layers.Dense(units=embedding_dim)
        self.value_dense = layers.Dense(units=embedding_dim)
        self.output_dense = layers.Dense(units=embedding_dim)

    def __attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(d_k)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output

    def __separate_head(self, matrix, batch_size):
        # Reshape: [-1, seq_len, embedding_dim] -> # [-1, seq_len, num_heads, project_dim].
        matrix = tf.reshape(matrix, shape=(batch_size, -1, self.num_heads, self.project_dim))
        return tf.transpose(matrix, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        query, key, value = self.query_dense(x), self.key_dense(x), self.value_dense(x)  # [-1, seq_len, embedding_dim]
        query, key, value = self.__separate_head(query, batch_size), self.__separate_head(key, batch_size), \
                            self.__separate_head(value, batch_size)  # [-1, num_heads, seq_len, project_dim]
        attention = self.__attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # [-1, seq_len, num_heads, project_dim]
        concat_attention = tf.reshape(attention, shape=(batch_size, -1, self.embedding_dim))
        output = self.output_dense(concat_attention)
        return output


class TransformerEncoder(layers.Layer):
    def __init__(self, embedding_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, num_heads)
        self.feed_forward = keras.Sequential(
            [layers.Dense(units=feed_forward_dim, activation='relu'), layers.Dense(units=embedding_dim)]
        )
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate=dropout_rate)
        self.dropout2 = layers.Dropout(rate=dropout_rate)

    def call(self, x):
        # Attention Block
        attention_output = self.attention(x)
        attention_output = self.dropout1(attention_output)
        attention_output = self.layer_norm1(x + attention_output)
        # Feed Forward Block
        feed_forward_output = self.feed_forward(attention_output)
        feed_forward_output = self.dropout2(feed_forward_output)
        output = self.layer_norm2(attention_output + feed_forward_output)
        return output


class TransformerClassifier(object):
    def __init__(self, classes, vocab_size, config, pretrained_embedding):
        self.classes, self.num_classes = classes, len(classes)
        self.vocab_size = vocab_size
        self.config = config
        self.pretrained_embedding = pretrained_embedding
        self.model = self.__create_model()

    def fit(self, x_train, y_train, x_val, y_val):
        if x_val is not None:
            self.model.fit(
                x=x_train, y=y_train,
                epochs=self.config['epochs'],
                verbose=True,
                validation_data=(x_val, y_val),
                batch_size=self.config['batch_size'])
        else:
            self.model.fit(
                x=x_train, y=y_train,
                epochs=self.config['epochs'],
                verbose=True,
                batch_size=self.config['batch_size'])

    def predict(self, x_test):
        return self.model.predict(x_test) >= 0.5

    def predict_prob(self, x_test):
        return self.model.predict(x_test)

    def __create_model(self):
        self.embedding_dim, self.num_heads, self.feed_forward_dim = \
            self.config['embedding_dim'], self.config['num_heads'], self.config['feed_forward_dim']
        self.dropout_rate = self.config['dropout_rate']
        self.maxlen = self.config['maxlen']

        embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size,
                                                    self.embedding_dim, self.pretrained_embedding)
        transformer_encoder = TransformerEncoder(self.embedding_dim, self.num_heads, self.feed_forward_dim,
                                                 dropout_rate=self.dropout_rate)

        inputs = layers.Input(shape=(self.maxlen, ))
        embedding_output = embedding_layer(inputs)
        transformer_output = transformer_encoder(embedding_output)

        outputs = layers.GlobalAveragePooling1D()(transformer_output)
        outputs = layers.Dropout(self.dropout_rate)(outputs)

        outputs = layers.Dense(units=64, activation='relu')(outputs)
        outputs = layers.Dropout(self.dropout_rate)(outputs)

        outputs = layers.Dense(units=self.num_classes, activation=None)(outputs)
        outputs = layers.Dense(units=self.num_classes, activation='sigmoid')(outputs)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        return model





