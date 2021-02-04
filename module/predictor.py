import csv


class Predictor(object):
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred

    def predict_prob(self, x_test):
        y_prob_pred = self.model.predict_prob(x_test)
        return y_prob_pred

    def save_result(self, test_ids, y_prob_pred):
        with open(self.config['output_path'], 'w') as output_csv_file:
            csv_writer = csv.writer(output_csv_file)
            header = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            csv_writer.writerow(header)
            for test_id, pred in zip(test_ids, y_prob_pred.tolist()):
                csv_writer.writerow([test_id] + pred)