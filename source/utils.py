from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
from joblib import dump
from scipy.stats import skew
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PowerTransformer
from imblearn.over_sampling import SMOTE
from source.classification import *
from source.clustering import clustering

def format_time(seconds):
    # Convert seconds to timedelta
    td = timedelta(seconds=seconds)
    # Format timedelta as HH:MM:SS
    return str(td)

def preprocess_data(data):
    # Import the churn_data dataset
    clean_data = data.drop(data.columns[[0, 21, 22]], axis=1)
    
    # Convert the 'Cluster_Label' column to object type
    clean_data['Cluster_Label'] = clean_data['Cluster_Label'].astype(object)

    # Normalize skewed numerical columns
    numerical_columns = clean_data.select_dtypes(include=['int', 'float']).columns
    skewed_cols = clean_data[numerical_columns].apply(lambda x: skew(x))
    skewed_cols = skewed_cols[skewed_cols > 0.5].index

    # Apply Yeo-Johnson power transform to skewed columns
    power_transformer = PowerTransformer(method='yeo-johnson')
    clean_data[skewed_cols] = power_transformer.fit_transform(clean_data[skewed_cols])

    # Standard scale the numerical columns
    scaler = MinMaxScaler()
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

def run_model(model_func, model_name, X_train, X_test, y_train, y_test):
    # Record start time
    start_time = time.time()

    # Run the specified model
    model = model_func(X_train, X_test, y_train, y_test)

    # Unpack model output
    y_pred, y_true = model

    # Evaluate the model
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    # Calculate run time
    end_time = time.time()
    run_time_seconds = end_time - start_time
    run_time_formatted = f"{run_time_seconds:.2f}"

    # Save the trained model
    model_filename = f"./files/models/{model_name}.pkl"
    dump(model, model_filename)

    # Log the performance metrics to event log
    event_log = pd.DataFrame({
        'Execution_Date': [datetime.now()],
        'Model': [model_name],
        'Run_Time': [run_time_seconds],  # Save run_time in seconds
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1],
        'Accuracy': [accuracy],
        'ROC_AUC': [roc_auc],
    })

    event_log.to_csv('./files/modeling_event_log.csv', mode='a', header=False, index=False)

    # Format metrics for terminal output
    precision_percent = f"{precision * 100:.2f}%"
    recall_percent = f"{recall * 100:.2f}%"
    f1_percent = f"{f1 * 100:.2f}%"
    accuracy_percent = f"{accuracy * 100:.2f}%"
    roc_auc_percent = f"{roc_auc * 100:.2f}%"

    print("Model:", model_name)
    print("Run Time:", run_time_formatted)
    print("Precision:", precision_percent)
    print("Recall:", recall_percent)
    print("F1 Score:", f1_percent)
    print("Accuracy:", accuracy_percent)
    print("ROC AUC:", roc_auc_percent)

def main():
    # Load the dataset
    raw_data = pd.read_csv('./files/raw_BankChurners.csv')
    clustered_data = clustering(raw_data)

    # Preprocess the data
    X, y = preprocess_data(clustered_data)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Prompt the user to select a model
    print("Which model would you like to run?")
    print("1. Logistic Regression")
    print("2. Naive Bayes")
    print("3. K-Nearest Neigbor")
    print("4. Decision Tree")
    print("5. Random Forest")
    print("6. Support Vector Machine")
    print("7. Ada Boost")
    print("8. Gradient Boosting")
    print("9. Neural Network")

    model_choice = input("Enter the number corresponding to your choice: ")

    if model_choice == '1':
        run_model(logistic_regression, 'Logistic Regression', X_train, X_test, y_train, y_test)
    elif model_choice == '2':
        run_model(naive_bayes, 'Naive Bayes', X_train, X_test, y_train, y_test)
    elif model_choice == '3':
        run_model(k_nearest_neighbors, 'K-Nearest Neighbor', X_train, X_test, y_train, y_test)
    elif model_choice == '4':
        run_model(decision_tree, 'Decision Tree', X_train, X_test, y_train, y_test)
    elif model_choice == '5':
        run_model(random_forest, 'Random Forest', X_train, X_test, y_train, y_test)
    elif model_choice == '6':
        run_model(support_vector_machine, 'Support Vector Machine', X_train, X_test, y_train, y_test)
    elif model_choice == '7':
        run_model(ada_boost, 'Ada Boost', X_train, X_test, y_train, y_test)
    elif model_choice == '8':
        run_model(gradient_boosting, 'Gradient Boosting', X_train, X_test, y_train, y_test)
    elif model_choice == '9':
        run_model(neural_network, 'Neural Network', X_train, X_test, y_train, y_test)
    else:
        print("Invalid choice. Please enter a valid number.")