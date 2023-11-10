# Group 31
# Kushagra Tomar
# Shyama Goel

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Function definitions

def load_data(train_path, test_path):
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"An error occurred: {e}")
        return None, None


def prepare_data(train_data, test_data, feature_columns, label_column, id_column):
    X = train_data[feature_columns]
    y = train_data[label_column]
    X_test = test_data[feature_columns]
    test_ids = test_data[id_column]
    return X, y, X_test, test_ids


def feature_selection_and_standardization(X, y, X_test):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    xgb_for_fs = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_for_fs.fit(X_scaled, y)

    selector = SelectFromModel(xgb_for_fs, prefit=True, threshold='median')
    X_selected = selector.transform(X_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    return X_selected, X_test_selected, selector


def tune_hyperparameters(X, y):
    params = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 400],
        'colsample_bytree': [0.5, 0.7, 0.9, 1],
        'subsample': [0.5, 0.7, 0.9, 1],
        'gamma': [0, 0.1, 0.2, 0.3]
    }

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42)
    strat_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_iter_search = 50
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=params,
        n_iter=n_iter_search,
        scoring='roc_auc',
        n_jobs=-1,
        cv=strat_k_fold,
        verbose=3,
        random_state=42
    )
    random_search.fit(X, y)
    return random_search.best_estimator_

def evaluate_model(model, X, y):
    y_pred = model.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_pred)

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)

def main(train_data_path, test_data_path):
    # Loading the datasets from the argument
    train_data, test_data = load_data(train_data_path, test_data_path)
    if train_data is not None and test_data is not None:

        # Preparing data
        feature_columns = train_data.columns.drop('Labels')
        X, y, X_test, test_ids = prepare_data(train_data, test_data, feature_columns, 'Labels', 'ID')

        # Feature selection and standardization
        X_selected, X_test_selected, selector = feature_selection_and_standardization(X, y, X_test)

        # Splitting data for validation
        X_train_selected, X_val_selected, y_train, y_val = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )

        # Model training and hyperparameter tuning
        best_xgb = tune_hyperparameters(X_train_selected, y_train)

        # Model evaluation
        val_auc = evaluate_model(best_xgb, X_val_selected, y_val)
        print(f"Validation AUC: {val_auc}")

        # Save the best model
        save_model(best_xgb, 'best_xgb_model.joblib')

        # Predict on test data
        test_predictions = best_xgb.predict_proba(X_test_selected)[:, 1]

        # Save the predictions
        submission = pd.DataFrame({
            'ID': test_ids,
            'Labels': test_predictions
        })
        submission.to_csv('submission333.csv', index=False)
        print("Submission file created successfully!")
    else:
        print("Data loading failed.")

#final execution

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate an XGBoost model for binary classification.')
    parser.add_argument('train_data', type=str, help='Path to the training data file (e.g., kaggle_train.csv)')
    parser.add_argument('test_data', type=str, help='Path to the test data file (e.g., kaggle_test.csv)')
    args = parser.parse_args()
    main(args.train_data, args.test_data)
