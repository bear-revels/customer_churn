# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc

# Encoding dictionaries
encoding_dicts = {
    'Attrition_Flag': {'Existing Customer': 0, 'Attrited Customer': 1},
    'Gender': {'M': 1, 'F': 0},
    'Education_Level': {'High School': 1, 'Graduate': 3, 'Uneducated': 0, 'Unknown': -1, 'College': 2, 'Post-Graduate': 4, 'Doctorate': 5},
    'Marital_Status': {'Married': 1, 'Single': 0, 'Unknown': -1, 'Divorced': 2},
    'Income_Category': {'$60K - $80K': 2, 'Less than $40K': 0, '$80K - $120K': 3, '$40K - $60K': 1, '$120K +': 4, 'Unknown': -1},
    'Card_Category': {'Blue': 0, 'Gold': 2, 'Silver': 1, 'Platinum': 3}
}

def label_encode_columns(data, encoding_dicts):
    encoded_data = data.copy()
    for column, encoder_dict in encoding_dicts.items():
        encoded_data[column] = encoded_data[column].map(encoder_dict)
    return encoded_data

def random_forest_classifier(data):
    # Import the churn_data dataset
    data = data.iloc[:, :-2]
    data = label_encode_columns(data, encoding_dicts)

    # Count the number of '-1' values in each row and create a new column
    data['Missing_Values_Count'] = (data == -1).sum(axis=1)

    # Split the data into features and target variable
    X = data.drop(columns=['Attrition_Flag'])
    y = data['Attrition_Flag']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier with cost-sensitive learning
    classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
    classifier.fit(X_train, y_train)

    # Predict probabilities for test data
    y_pred_proba = classifier.predict_proba(X_test)

    # Adjust the threshold to improve recall for attrited customers
    threshold = 0.4  # Adjust this threshold as needed
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)

    # Evaluate the model on test data
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot ROC curve
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Load and model the dataset
churn_data = pd.read_csv('./files/raw_BankChurners.csv')
random_forest_classifier(churn_data)