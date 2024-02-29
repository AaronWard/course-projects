# library doc string
"""
Library for creating models for prediction customer churn.
Includes utilities for feature engineering, performing EDA
and evaluation report images.

The modelling utilities are defined in a class structure.

Author: Aaron Ward
Date: December 2021
"""

# import libraries
import os
import time
import logging
from abc import ABC, abstractmethod

import shap
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

from constants import keep_cols, cat_columns

sns.set()

logging.basicConfig(
    filename=f"./logs/churn_library_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    try:
        print(f'Importing data from {pth}')
        return pd.read_csv(pth, index_col=0)
    except FileNotFoundError as err:
        logging.error(f"ERROR: file {pth} not found")
        raise err


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            outputs visualizations within the image_path
    '''
    image_path = "./images/eda"

    plt.figure(figsize=(20, 10))
    plt.title('Histogram of churn/non-churn members')
    data_frame['Churn'].hist()
    plt.savefig(os.path.join(image_path, 'churn_hist.png'))

    plt.figure(figsize=(20, 10))
    plt.title('Histogram of customers age')
    data_frame['Customer_Age'].hist()
    plt.savefig(os.path.join(image_path, 'age_hist.png'))

    plt.figure(figsize=(20, 10))
    plt.title('Volume of marital status')
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(image_path, 'marital_status_bar.png'))

    plt.figure(figsize=(20, 10))
    sns.displot(data_frame['Total_Trans_Ct'], kde=True, height=10)
    plt.savefig(os.path.join(image_path, 'trans_distribution.png'))

    plt.figure(figsize=(20, 10))
    plt.title('Data heatmap')
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(image_path, 'corr_heatmap.png'))


def encoder_helper(data_frame, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming
                      variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''

    for col in category_lst:
        groups = data_frame.groupby(col).mean()['Churn']
        data_frame[f'{col}_Churn'] = data_frame[col].apply(lambda x: groups[x])

    return data_frame


def perform_feature_engineering(data_frame):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    x = data_frame[keep_cols]
    y = data_frame['Churn']

    return train_test_split(x, y, test_size=0.3, random_state=42)


#########################################################################


def classification_report_image(models):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            models: list of model
            y_train: training response values
            y_test:  test response values

    output:
             None
    '''
    image_path = './images/results/'

    print(f'Generating ROC curve for {len(models)} models')
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    [plot_roc_curve(model.load_model(), model.x_test, model.y_test,
                    ax=ax, alpha=0.8) for model in models]
    plt.savefig(os.path.join(image_path, "classification_report.png"))
    plt.close()

    for model in models:
        print(f'Generating classification report for {model.get_name()}')
        plt.rc('figure', figsize=(5, 5))
        plt.text(
            x=0.01,
            y=1,
            s=str(f'{model.get_name()} Train'),
            fontdict={
                'fontsize': 10})
        plt.text(
            x=0.01,
            y=0.5,
            s=str(
                classification_report(
                    model.y_train,
                    model.y_train_preds)),
            fontdict={
                'fontsize': 10},
            fontproperties='monospace')

        plt.text(
            x=0.01,
            y=0.45,
            s=str(f'{model.get_name()} Test'),
            fontdict={
                'fontsize': 10})
        plt.text(
            x=0.01,
            y=0.001,
            s=str(
                classification_report(
                    model.y_test,
                    model.y_test_preds)),
            fontdict={
                'fontsize': 10},
            fontproperties='monospace')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                image_path,
                f"{model.get_name()}_classification_report.png"))
        plt.close()

    print(f'\nImages save to {image_path}')


def feature_importance_plot(model, output_pth=None):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    if output_pth is None:
        output_pth = './images/feature_importances'

    loaded_model = model.load_model()

    try:
        explainer = shap.TreeExplainer(loaded_model.best_estimator_)
        shap_values = explainer.shap_values(model.x_test)
        plt.figure(figsize=(20, 5))
        shap.summary_plot(
            shap_values,
            model.x_test,
            plot_type="bar",
            show=False)
        plt.savefig(
            os.path.join(
                output_pth,
                f"{model.get_name()}_shap_summary.png"))
        plt.close()

        # Calculate feature importances
        importances = loaded_model.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]
        names = [model.x_train.columns[i] for i in indices]
        plt.figure(figsize=(20, 5))
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(model.x_train.shape[1]), importances[indices])
        plt.xticks(range(model.x_train.shape[1]), names, rotation=90)
        plt.savefig(
            os.path.join(
                output_pth,
                f"{model.get_name()}_feature_importances.png"))
        plt.close()

        print(f"Images saved to {output_pth}")
    except AttributeError:
        """
        AttributeError: 'LogisticRegression' object has no attribute 'best_estimator_'
        """
        print(
            f"Cant perform feature importance analysis using Shap with {model.get_name()}")
        logging.error(
            f"ERROR: failed to perform feature engineering with {model.get_name()}")

#########################################################################


class Model(ABC):
    """
    This is the blueprint for a Model class.

    functionality:
    --------------
    - get_name(): returns the name of the instance you set.
    - train_model(): abstract method to train different models
    - save_model(): saves the model to a .pkl file
    - load_model(): loads the model .pkl file
    - get_train_test_results(): returns a classification report
    """

    def __init__(self):
        self.model_name = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_preds = None
        self.y_test_preds = None
        self.model = None
        self.model_path = None

    def get_name(self):
        '''
        Return the name of the model
        '''
        return self.model_name

    @abstractmethod
    def train_model(self):
        '''
        Abstract method for Model class.
        Is used for training the models in each derived class
        '''
        pass

    def save_model(self):
        '''
        Function for saving model
        - models should be stored in ./models/ folder
        '''
        if self.model_path is None:
            self.model_path = f'./models/'

        if not os.path.exists(self.model_path):
            print(f'Making now directory: {self.model_path}')
            os.makedirs(self.model_path)

        if self.model_name is None:
            self.model_name = f'{self.get_name()}'

        print(f"Saving model to {self.model_path}")
        joblib.dump(
            self.model,
            os.path.join(
                self.model_path,
                f"{self.model_name}.pkl"))

    def load_model(self):
        '''
        Function for loading saved model
        - models should be stored in ./models/ folder
        '''
        if self.model_path is None:
            self.model_path = f'./models/'
        if self.model_name is None:
            self.model_name = f'{self.get_name()}'

        print(f"Loading model from {self.model_path}")
        return joblib.load(
            os.path.join(
                self.model_path,
                f"{self.model_name}.pkl"))

    def get_train_test_results(self):
        '''
        use sklearn.metrics to get classification_report
        '''
        print(f'{self.get_name()} results')
        print('test results')
        print(classification_report(self.y_test, self.y_test_preds))
        print('train results')
        print(classification_report(self.y_train, self.y_train_preds))
        return (
            'test results',
            classification_report(
                self.y_test,
                self.y_test_preds))


class RandomForestModel(Model):
    """
    Random Forest Model, which inherits from the Model class

    """

    def __init__(self, x_train, x_test, y_train, y_test,
                 model_name='Random Forest', model_path='./models/'):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_preds = None
        self.y_test_preds = None
        self.model_name = model_name
        self.model_path = model_path
        self.model = RandomForestClassifier(random_state=42)

    def train_model(self):
        print(f'Training model: {self.get_name()}')
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }

        self.model = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5)
        self.model.fit(self.x_train, self.y_train)

        self.y_train_preds = self.model.predict(self.x_train)
        self.y_test_preds = self.model.predict(self.x_test)


class LinearRegressionModel(Model):
    '''
    Linear Regression Model

    '''

    def __init__(self, x_train, x_test, y_train, y_test,
                 model_name='Linear Regression', model_path='./models/'):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_train_preds = None
        self.y_test_preds = None
        self.model_name = model_name
        self.model_path = model_path
        self.model = LogisticRegression(max_iter=1000)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

        self.y_train_preds = self.model.predict(self.x_train)
        self.y_test_preds = self.model.predict(self.x_test)


if __name__ == "__main__":
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    DATA = import_data("./data/bank_data.csv")
    DATA['Churn'] = DATA['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(DATA)
    DATA = encoder_helper(DATA, cat_columns)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(DATA)

    RF_MODEL = RandomForestModel(X_TRAIN,
                                 X_TEST,
                                 Y_TRAIN,
                                 Y_TEST,
                                 model_name="rf_test",
                                 model_path='./test_output/')

    LR_MODEL = LinearRegressionModel(X_TRAIN,
                                     X_TEST,
                                     Y_TRAIN,
                                     Y_TEST,
                                     model_name="lr_test",
                                     model_path='./test_output/')
    LR_MODEL.train_model()
    LR_MODEL.get_train_test_results()
    LR_MODEL.save_model()
    classification_report_image(models=[LR_MODEL])
    feature_importance_plot(LR_MODEL)
