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


def load_adult(adult_names, rootdir=rootdir, split='train'):
    csv_name = 'adult.data' if split == 'train' else 'adult.test'
    csv_path = os.path.join(rootdir, csv_name)
    assert os.path.exists(csv_path)
    adult_data = pd.read_csv(csv_path, names=adult_names + ['salary'], index_col=False)
    return adult_data

def get_faulty_col_names(df):
    faulty_cols = []
    
#     Check NA values
    for col_name in df.columns.tolist():
        if df[col_name].isna().values.any() is True:
            faulty_cols.append(col_name)

#     Check for ' ?'
        if ' ?' in df[col_name].unique().tolist():
            if col_name not in faulty_cols:
                faulty_cols.append(col_name)
            print(f"Number of ' ?' in {col_name}: {df[col_name].value_counts()[' ?']}")
    
    return faulty_cols

def replace_with_mode(df, faulty_cols):
    
    df_without_QM = df.copy()
    for col in faulty_cols:
        df_without_QM = df_without_QM[getattr(df, col) != ' ?']
    
    r_df = df.copy()
    mapping = {}
    for col in faulty_cols:
        mapping[col] = df_without_QM[col].mode().values[0]
        r_df[col] = r_df[col].replace(' ?', mapping[col])
    return r_df, mapping

def replace_with_mapping(df, faulty_cols, mapping):
    r_df = df.copy()
    for col in faulty_cols:
        r_df[col] = r_df[col].replace(' ?', mapping[col])
    return r_df

def get_binary_labels_for_salary(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    assert isinstance(df, pd.DataFrame)
    salary_map = {' <=50K': 0, ' <=50K.': 0, ' >50K': 1, ' >50K.': 1}
    df['salary'] = df['salary'].map(salary_map)
    
    return df['salary'].tolist()

def generate_cross_validation(n_folds, X, y):
    samples = len(X)
    elements_in_fold = samples // n_folds
    indices = [0]
    while indices[-1] + elements_in_fold <= samples - samples % n_folds:
        indices.append(indices[-1] + elements_in_fold)
    indices[-1] = samples
    cross_val_sets = [[indices[i], indices[i+1]] for i in range(len(indices)-1)]
    for val_set in cross_val_sets:
        begin, end = val_set[0], val_set[1]
        train_X_fold = X[0:begin].append(X[end:])
        validation_X_fold = X[begin:end]
        train_y_fold = y[0:begin] + y[end:]
        validation_y_fold = y[begin:end]
        yield train_X_fold, train_y_fold, validation_X_fold, validation_y_fold

def knn_fit(X_train, y_train, X_val, y_val, one_hot, knn=3):
    neigh = KNeighborsClassifier(n_neighbors=knn)
    neigh.fit(one_hot.transform(X_train), y_train)
    pred_y = neigh.predict(OH_enc.transform(X_val))
    acc = get_accuracy(y_val, pred_y)
    return neigh, acc

def eval_classifier(X, y, one_hot, classifier):
    pred_y = classifier.predict(one_hot.transform(X))
    acc = get_accuracy(y, pred_y)
    return acc

def fit_classifier(X_train, y_train, X_val, y_val, one_hot, classifier_model, **kwargs):
    classifier = classifier_model(**kwargs)
    classifier.fit(one_hot.transform(X_train), y_train)
    acc = eval_classifier(X_val, y_val, one_hot, classifier)
    return classifier, acc

def get_accuracy(y_label, y_pred):
    assert len(y_label) == len(y_pred)
    acc = 100 * sum(y_label == y_pred) / len(y_label)
    return acc
