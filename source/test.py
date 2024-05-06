from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import make_scorer, classification_report
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Custom transformer to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=[self.dtype])

# Custom transformer to perform Yeo-Johnson transformation
class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.nan_to_num(np.log1p(X))

# Load data and identify column types
# Assume you have loaded your data into a DataFrame called "data"
data = pd.read_csv('./files/raw_BankChurners.csv').iloc[:, 1:-2]
target_variable = data['Attrition_Flag']
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
categorical_features = data.select_dtypes(include=['object']).columns

# Define preprocessing steps for numerical and categorical columns
numerical_pipeline = Pipeline([
    ('selector', DataFrameSelector(dtype='float64')), # Select numerical columns
    ('yeo_johnson', FunctionTransformer(YeoJohnsonTransformer())), # Apply Yeo-Johnson transformation
    ('scaler', MinMaxScaler()) # Scale data
])

categorical_pipeline = Pipeline([
    ('selector', DataFrameSelector(dtype='object')), # Select categorical columns
    ('one_hot_encoder', OneHotEncoder(drop=None)), # One-hot encode categorical data
    ('scaler', MinMaxScaler()) # Scale data
])

# Combine numerical and categorical pipelines using FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
    ('numerical_pipeline', numerical_pipeline),
    ('categorical_pipeline', categorical_pipeline)
])

# PCA for feature selection and dimensionality reduction
pca = PCA()

# KMeans clustering
kmeans = KMeans()

# XGBoost classifier
xgb = XGBClassifier()

# Define the main pipeline with SMOTE
pipeline = ImbPipeline([
    ('preprocess', preprocess_pipeline),
    ('smote', SMOTE(random_state=42)),  # Integrate SMOTE into the pipeline
    ('pca', pca),
    ('kmeans', kmeans),
    ('xgb', xgb)
])

# Define parameter grid for hyperparameter optimization
param_grid = {
    'pca__n_components': [5, 10, 15], # Adjust number of PCA components
    'kmeans__n_clusters': [2, 3, 4], # Adjust number of clusters
    'xgb__n_estimators': [100, 200, 300], # Number of boosting rounds
    'xgb__max_depth': [3, 4, 5], # Maximum tree depth
}

# Perform grid search cross-validation with standard scoring technique
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1') # or any other appropriate metric
grid_search.fit(data, target_variable)

# Get best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Fit pipeline with best parameters
pipeline.set_params(**best_params)
pipeline.fit(data, target_variable)

# Predict clusters
clusters = pipeline.named_steps['kmeans'].labels_

# Get predictions for each cluster
cluster_predictions = {}
for cluster_label in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster_label)[0]
    cluster_data = data.iloc[cluster_indices]
    cluster_target = target_variable.iloc[cluster_indices]
    predictions = pipeline.predict(cluster_data)
    cluster_predictions[cluster_label] = (cluster_target, predictions)

# Print classification report for each cluster
for cluster_label, (true_labels, predicted_labels) in cluster_predictions.items():
    print(f"Classification Report for Cluster {cluster_label}:")
    print(classification_report(true_labels, predicted_labels))