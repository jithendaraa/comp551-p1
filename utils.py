import os
import pandas as pd

rootdir = 'Adult dataset'

adult_names = ["age",
"workclass",
"fnlwgt",
"education",
"education-num",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"capital-gain",
"capital-loss",
"hours-per-week",
"native-country"]

def load_adult(rootdir=rootdir):
    csv_path = os.path.join(rootdir, 'adult.data')
    assert os.path.exists(csv_path)
    adult_data = pd.read_csv(csv_path, names=adult_names + ['salary'], index_col=False)
    return adult_data

def check_data(df):
    pass
