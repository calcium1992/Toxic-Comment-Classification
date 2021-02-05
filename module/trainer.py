from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from module.model import NaiveBayes, CNN


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
        else:
            model_name = self.config['model_name']
            self.logger.warning(f'Model {model_name} is not supported yet.')

    @staticmethod
    def __evaluate(y_val, y_pred):
        accuracy = accuracy_score(y_val, y_pred)
        cls_report = classification_report(y_val, y_pred, zero_division=1)
        return accuracy, cls_report