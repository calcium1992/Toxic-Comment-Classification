from sklearn.metrics import accuracy_score, f1_score, classification_report
from module.model import NaiveBayes, CNN, RNN, TransformerClassifier


class Trainer(object):
    def __init__(self, config, logger, preprocessor):
        self.config = config
        self.logger = logger
        self.preprocessor = preprocessor
        self.__create_model()

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit(x_train, y_train, x_val, y_val)
        return self.model

    def validate(self, x_val, y_val):
        y_pred = self.model.predict(x_val)
        return Trainer.__evaluate(y_val, y_pred)

    def __create_model(self):
        if self.config['model_name'] == 'naivebayes':
            self.model = NaiveBayes(self.preprocessor.classes)
        elif self.config['model_name'] == 'cnn':
            self.model = CNN(self.preprocessor.classes, self.preprocessor.vocab_size, self.config)
        elif self.config['model_name'] == 'cnnglove':
            if self.preprocessor.pretrained_embedding is not None:
                self.model = CNN(self.preprocessor.classes, self.preprocessor.vocab_size, self.config,
                                 pretrained_embedding=self.preprocessor.pretrained_embedding)
            else:
                self.logger.warning(f'Pretrained embedding is not available.')
        elif self.config['model_name'] == 'rnnglove':
            if self.preprocessor.pretrained_embedding is not None:
                self.model = RNN(self.preprocessor.classes, self.preprocessor.vocab_size, self.config,
                                 pretrained_embedding=self.preprocessor.pretrained_embedding)
            else:
                self.logger.warning(f'Pretrained embedding is not available.')
        elif self.config['model_name'] == 'transformer':
            if self.preprocessor.pretrained_embedding is not None:
                self.model = TransformerClassifier(self.preprocessor.classes, self.preprocessor.vocab_size, self.config,
                                                   pretrained_embedding=self.preprocessor.pretrained_embedding)
            else:
                self.logger.warning(f'Pretrained embedding is not available.')
        else:
            model_name = self.config['model_name']
            self.logger.warning(f'Model {model_name} is not supported yet.')

    @staticmethod
    def __evaluate(y_val, y_pred):
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='samples', zero_division=1)
        cls_report = classification_report(y_val, y_pred, zero_division=1)
        return accuracy, f1, cls_report