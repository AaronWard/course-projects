"""
Script for unit testing the functionality in churn_library.py
Test by running 'ipython churn_script_logging_and_tests.py'

Author: Aaron Ward
Date: December 2021
"""

import os
import time
import logging
import churn_library as cls
from constants import cat_columns

logging.basicConfig(
    filename=f"./logs/test_churn_library__{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import_data(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return data_frame


def test_perform_eda(perform_eda, data_frame):
    '''
    test perform eda function

    '''

    perform_eda(data_frame)
    eda_image_path = "./images/eda"

    # Checking if the list is empty or not
    try:
        # Getting the list of directories
        assert len(os.listdir(eda_image_path)) != 0
        logging.info("SUCCESS: test_perform_eda passed")
    except AssertionError as err:
        logging.error(
            "ERROR: Images not found in EDA folder after running perform_eda()")
        raise err


def test_encoder_helper(encoder_helper, data_frame):
    '''
    test encoder helper
    '''
    num_cols = len(data_frame.columns)
    data_frame = encoder_helper(data_frame, cat_columns)

    try:
        # Test length of returned columns
        assert len(data_frame.columns) == len(cat_columns) + num_cols
        logging.info("SUCCESS: test_encoder_helper passed")
        return data_frame
    except AssertionError as err:
        logging.error(
            "ERROR: encoder_helper() return incorrect number of columns.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, data_frame):
    '''
    test perform_feature_engineering
    '''
    data_list = []
    [data_list.append(item)
     for item in perform_feature_engineering(data_frame)]

    # ensure no empty data is returned
    try:
        for item in data_list:
            assert item.shape[0] > 0
        logging.info("SUCCESS: test_perform_feature_engineering passed")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "The four objects that should be returned were not.")
        raise err

    return data_list[0], data_list[1], data_list[2], data_list[3]


def test_train_model(train_model, model):
    '''
    test train_models
    '''
    train_model()
    try:
        model.save_model()
        assert os.path.exists(os.path.join(model.model_path,
                                           f"{model.model_name}.pkl"))
        logging.info("SUCCESS: Successfully trained model and saved")
    except AssertionError:
        logging.error("ERROR: Model failed to train correctly, couldn't load.")


def test_save_model(save_model, model):
    '''
    Test save_model()
    '''
    save_model()
    try:
        expected_file = os.path.join(
            model.model_path, f"{model.model_name}.pkl")
        assert os.path.exists(expected_file)
        logging.info(f"SUCCESS:: Model was saved to {expected_file}")
    except FileNotFoundError:
        logging.error(f"ERROR: Model was not saved to {expected_file}")


def test_feature_importance_plot(feature_importance_plot, model):
    '''
    Test to see feature importances are saved.
    '''

    feature_importance_plot(model)
    try:
        expected_file = os.path.join(
            './images/feature_importances',
            f"{model.get_name()}_feature_importances.png")
        assert os.path.exists(expected_file)
        logging.info(
            f"SUCCESS: Feature importance plot was saved to {expected_file}")
    except FileNotFoundError:
        logging.error(
            f"ERROR: Feature importance plot was not saved to {expected_file}")


def test_classification_report_image(classification_report_image, models=None):
    '''
    Tests for seeing if classification report images
    was generated.
    '''

    classification_report_image(models=models)
    try:
        expected_file = os.path.join(
            "./images/results/",
            "classification_report.png")
        assert os.path.exists(expected_file)
        logging.info(
            f"SUCCESS: Classification Report was saved to {expected_file}")
    except FileNotFoundError:
        logging.error(
            f"ERROR: Classification Report  was not saved to {expected_file}")


if __name__ == "__main__":
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    DATA_FRAME = test_import_data(cls.import_data)
    DATA_FRAME['Churn'] = DATA_FRAME['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    test_perform_eda(cls.perform_eda, DATA_FRAME)
    DATA_FRAME = test_encoder_helper(cls.encoder_helper, DATA_FRAME)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA_FRAME)

    # Define model instances
    RF_MODEL = cls.RandomForestModel(X_TRAIN,
                                     X_TEST,
                                     Y_TRAIN,
                                     Y_TEST,
                                     model_name="test_rf_model",
                                     model_path="./test_output/")

    LR_MODEL = cls.LinearRegressionModel(X_TRAIN,
                                         X_TEST,
                                         Y_TRAIN,
                                         Y_TEST,
                                         model_name="test_lr_model",
                                         model_path="./test_output/")

    test_train_model(RF_MODEL.train_model, RF_MODEL)
    test_train_model(LR_MODEL.train_model, LR_MODEL)
    test_save_model(LR_MODEL.save_model, LR_MODEL)
    test_save_model(LR_MODEL.save_model, LR_MODEL)
    test_classification_report_image(
        cls.classification_report_image, models=[
            RF_MODEL, LR_MODEL])
    test_feature_importance_plot(cls.feature_importance_plot, RF_MODEL)
