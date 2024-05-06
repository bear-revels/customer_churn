import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_curve, auc
from prince import MCA

# Load and model the dataset
churn_data = pd.read_csv('./files/raw_BankChurners.csv')

# Define numerical and categorical columns
numerical_cols = churn_data.select_dtypes(include=['number']).columns
categorical_cols = churn_data.select_dtypes(include=['object']).columns

# Define the preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Label encode the target column
target_encoder = LabelEncoder()

# Combine preprocessing steps with PCA and classifier
classifier_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=0.95)),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

def train_and_evaluate_pipeline(csv_file_path, classifier_pipeline):
    # Load and model the dataset
    data = pd.read_csv(csv_file_path)
    data.info()

    # Label encode the target column
    data['Attrition_Flag'] = target_encoder.fit_transform(data['Attrition_Flag'])

    # Split the data into features and target variable
    X = data.drop(columns=['Attrition_Flag'])
    y = data['Attrition_Flag']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the classifier
    classifier_pipeline.fit(X_train, y_train)

    # Predict probabilities for test data
    y_pred_proba = classifier_pipeline.predict_proba(X_test)

    # Adjust the threshold to improve recall for attrited customers
    threshold = 0.4  # Adjust this threshold as needed
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)

    # Evaluate the model on test data
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot ROC curve
    y_pred_proba = classifier_pipeline.predict_proba(X_test)[:, 1]
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

# Train and evaluate the pipeline
train_and_evaluate_pipeline('./files/raw_BankChurners.csv', classifier_pipeline)
