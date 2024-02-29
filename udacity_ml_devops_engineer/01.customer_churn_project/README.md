# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to predict customer churn using the `bank_data` dataset.



## Running Files

**Dependencies**
```python
shap
joblib
pandas
numpy==1.19.5
matplotlib
seaborn==0.11.0
scikit-learn==0.23.2
pylint
autopep8
```
Firstly, install the dependencies needed to run this codebase:

```powershell
pip install -r requirements.txt
```

**Modelling notebook**
- `churn_notebook.ipynb`
    - Loads in the bank data.
    - Creates a column called `Churn` which will act as out label.
    - Perform EDA and save the output to `./images/eda`.
    - Feature engineering: create additional features and split the data into training and testing sets.
    - Model training: train models, get the initial results and save the model to `./models`
    - Generate a classification report: Produce ROC diagrams and save to `./images/results/`
    - Explanability: generate a shap summary and feature importance diagram and save to `./images/feature_importances`

**Churn Modelling Library**
- `churn_library.py`
    - This file contains utilities for performing EDA, Feature engineering, modellign and generating reports.
    - Each model type is derived from the `Model` class, which is a blueprint for the how a model will act.
    - Each new model should have a `train_model()` function


### Testing
- `churn_script_logging_and_tests.py`
    - this file contains test functions to unit test each utility in `churn_library.py`.
    - You can run tests by running `ipython churn_script_logging_and_tests.py`
    
<hr>

**Written by:** Aaron Ward


