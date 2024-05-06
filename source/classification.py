from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import skew
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np

def preprocess_data(data):
    # Import the churn_data dataset
    clean_data = data.iloc[:, 1:-2]

    # Normalize skewed numerical columns
    numerical_columns = clean_data.select_dtypes(include=['int', 'float']).columns
    skewed_cols = clean_data[numerical_columns].apply(lambda x: skew(x))
    skewed_cols = skewed_cols[skewed_cols > 0.5].index
    clean_data[skewed_cols] = np.log1p(clean_data[skewed_cols])

    # Standard scale the numerical columns
    scaler = StandardScaler()
    clean_data[numerical_columns] = scaler.fit_transform(clean_data[numerical_columns])

    # Count the number of '-1' values in each row and create a new column
    clean_data['Missing_Values_Count'] = (clean_data == -1).sum(axis=1)

    # Label encode 'Attrition_Flag'
    label_encoder = LabelEncoder()
    clean_data['Attrition_Flag'] = label_encoder.fit_transform(clean_data['Attrition_Flag'])

    # One-hot encode the categorical columns
    categorical_columns = clean_data.select_dtypes(include=['object']).columns
    clean_data = pd.get_dummies(clean_data, columns=categorical_columns)

    # Split the data into features and target variable
    X = clean_data.drop(columns=['Attrition_Flag'])
    y = clean_data['Attrition_Flag']

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled

def logistic_regression(X_train, X_test, y_train, y_test):
    # Train a Logistic Regression model
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def random_forest(X_train, X_test, y_train, y_test):
    # Train a Random Forest model
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def support_vector_machine(X_train, X_test, y_train, y_test):
    # Train a Support Vector Machine model
    classifier = SVC(random_state=42, probability=True)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def gradient_boosting(X_train, X_test, y_train, y_test):
    # Train a Gradient Boosting model
    classifier = XGBClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def neural_network(X_train, X_test, y_train, y_test):
    # Train a Neural Network model
    classifier = MLPClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    # Train a K-Nearest Neighbors model
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def naive_bayes(X_train, X_test, y_train, y_test):
    # Train a Naive Bayes model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def decision_tree(X_train, X_test, y_train, y_test):
    # Train a Decision Tree model
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test

def ada_boost(X_train, X_test, y_train, y_test):
    # Train an Ensemble method (AdaBoost) model
    classifier = AdaBoostClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred = classifier.predict(X_test)

    return y_pred, y_test