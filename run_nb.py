import yaml
import logging
import argparse
from module import Preprocessor, Trainer, Predictor

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Process commandline')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default="INFO")
    args = parser.parse_args()

    # Logger
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=args.log_level)
    logger = logging.getLogger('global_logger')

    # Main
    with open(args.config, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)

            # Preprocessing
            preprocessor = Preprocessor(config=config['preprocessing'], logger=logger)
            x, y, x_train, y_train, x_val, y_val, x_test = preprocessor.process()
            print(x.shape, y.shape, x_train.shape, y_train.shape, x_test.shape)

            # Training
            trainer = Trainer(config=config['training'], logger=logger, preprocessor=preprocessor)
            trainer.fit(x_train, y_train, x_val, y_val)
            accuracy, cls_report = trainer.validate(x_val, y_val)
            logger.info(f"accuracy:{accuracy}")
            # logger.info("\n{}\n".format(cls_report))

            # Predicting
            final_model = trainer.fit(x, y, None, None)
            predictor = Predictor(config=config['predict'], logger=logger, model=final_model)
            y_prob_pred = predictor.predict_prob(x_test)
            predictor.save_result(preprocessor.test_ids, y_prob_pred)
        except yaml.YAMLError as err:
            logger.warning(f'Config file err: {err}')