import sys
import argparse
import csv
import joblib
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report

col_drops = ['ID', 'employment', 'extended_reason', 'zipcode']
encode = ['loan_duration', 'employment_length', 'race', 'reason_for_loan',
          'employment_verified', 'state', 'home_ownership_status', 'type_of_application']
nans = ['public_bankruptcies', 'fico_inquired_last_6mths', 'months_since_last_delinq', 'any_tax_liens']


def preprocess(file, file_type):
    data = pd.read_csv(file).drop(columns=col_drops)
    data = data.fillna('nan')
    if file_type == 'train':
        les = []
        for feature in encode:
            le = preprocessing.LabelEncoder()
            le.fit(data[feature])
            data[feature] = le.transform(data[feature])
            les.append(le)
        with open('encodings.pk', 'wb') as fout:
            pickle.dump(les, fout)
    else:
        les = pickle.load(open('encodings.pk', 'rb'))
        for i in range(len(les)):
            data[encode[i]] = les[i].transform(data[encode[i]])
    for i in range(len(nans)):
        data[nans[i]] = data[nans[i]].replace('nan', -1)
    features = len(data.columns) - 1
    x, y = data.iloc[:, :features], data.iloc[:, features]
    if file_type == 'train':
        scale = StandardScaler()
        scale.fit(x)
        x = scale.transform(x)
        with open('scale.pk', 'wb') as fout:
            pickle.dump(scale, fout)
    else:
        scale = pickle.load(open('scale.pk', 'rb'))
        x = scale.transform(x)
    x = np.nan_to_num(x.astype(np.float32))
    return x, y


def mcc(y_test, y_pred):
    y_test[y_test == 0] = -1
    y_test[y_test == 1] = +1
    y_pred[y_pred == 0] = -1
    y_pred[y_pred == 1] = +1
    acc = matthews_corrcoef(y_test, y_pred)
    print(f'Matthews Correlation Coefficient: {acc}')
    y_test[y_test == -1] = 0
    y_test[y_test == +1] = 1
    y_pred[y_pred == -1] = 0
    y_pred[y_pred == +1] = 1
    return acc


def gbc(X_train, Y_train):
    print("Training GBC...")
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=2)
    model.fit(X_train, Y_train)
    joblib.dump(model, "gbc_model.pkl")
    return model


def mlp(X_train, Y_train):
    print("Training MLP...")
    model = MLPClassifier(max_iter=600, alpha=0.05, hidden_layer_sizes=(100,),
                          learning_rate='constant', solver='adam')
    model.fit(X_train, Y_train)
    joblib.dump(model, "mlp_model.pkl")
    return model


def logistic(X_train, Y_train):
    print("Training LR...")
    model = LogisticRegression(penalty="l2", tol=0.001, C=0.1, fit_intercept=True,
                               solver="lbfgs", intercept_scaling=1, max_iter=1000,
                               class_weight={0: 1, 1: 1.25})
    model.fit(X_train, Y_train)
    joblib.dump(model, "lr_model.pkl")
    return model


def xgboost(X_train, Y_train):
    print("Training XGBoost...")
    model = xgb.XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.25,
                              min_child_weight=9, scale_pos_weight=1.25, subsample=0.8,
                              colsample_bytree=0.7, max_delta_step=1, gamma=0)
    model.fit(X_train, Y_train)
    joblib.dump(model, "xgb_model.pkl")
    return model


def rfc(X_train, Y_train):
    print("Training RFC...")
    model = RandomForestClassifier(max_depth=14)
    model.fit(X_train, Y_train)
    joblib.dump(model, "rfc_model.pkl")
    return model


def test_bagging(X):
    models = ["gbc_model.pkl", "mlp_model.pkl", "xgb_model.pkl"]
    bagged_predictions = []
    for m in models:
        model = joblib.load(m)
        bagged_predictions.append(model.predict(X))
    print("Predictions Complete")
    df = pd.DataFrame.from_records(bagged_predictions).T
    predictions = df.mode(axis=1).values.tolist()
    print("Predictions Bagged")
    ID = 1000000
    with open('predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'loan_paid'])
        for y in predictions:
            writer.writerow([ID, y[0]])
            ID += 1


def train_bagging(X, Y):
    train_df = pd.DataFrame(data=X)
    train_df['Y'] = Y.tolist()
    samples = []
    for i in range(5):
        samples.append(train_df.sample(frac=0.6, replace=True))
    n = len(train_df.columns) - 1
    gbc(samples[0].iloc[:, :n], samples[0].iloc[:, n])
    mlp(samples[1].iloc[:, :n], samples[1].iloc[:, n])
    logistic(samples[2].iloc[:, :n], samples[2].iloc[:, n])
    xgboost(samples[3].iloc[:, :n], samples[3].iloc[:, n])
    rfc(samples[4].iloc[:, :n], samples[4].iloc[:, n])


def run(arguments):
    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('-e', '--test', help='Test attributes (to predict)')
    group1.add_argument('-n', '--train', help='Train data')
    args = parser.parse_args(arguments)
    if args.train is not None:
        print(f"Training data file: {args.train}")
        x, y = preprocess(args.train, 'train')
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3)
        ros = RandomOverSampler()
        X_train, Y_train = ros.fit_resample(X_train, Y_train)
        model = logistic(X_train, Y_train)
        y_pred = model.predict(X_test)
        mcc(Y_test, y_pred)
        print(classification_report(Y_test, y_pred))
    elif args.test is not None:
        print(f"Prediction data file: {args.test}")
        x, y = preprocess(args.test, 'test')
        model = joblib.load("xgb_model.pkl")
        predictions = model.predict(x)
        ID = 1000000
        with open('predictions.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'loan_paid'])
            for i in predictions:
                writer.writerow([ID, i])
                ID += 1
    else:
        print("Error in argument")


if __name__ == "__main__":
    run(sys.argv[1:])
