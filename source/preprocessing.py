import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class LogTransformSkewedFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer for log transforming skewed features.

    Parameters:
    - skew_threshold: Threshold for skewness. Default is 0.6.
    """

    def __init__(self, skew_threshold=0.6):
        self.skew_threshold = skew_threshold
    
    def fit(self, X, y=None):
        """Fit method."""
        return self
    
    def transform(self, X):
        """Transform method."""
        skewed_features = X.apply(lambda x: x.skew())
        skewed_features = skewed_features[skewed_features > self.skew_threshold].index
        for feature in skewed_features:
            X[feature] = np.log1p(X[feature])
        return X

class LabelEncodeColumns(BaseEstimator, TransformerMixin):
    """
    Transformer for label encoding categorical columns.

    This class encodes specific columns with predefined dictionaries.

    No parameters are required for initialization.
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """Fit method."""
        return self
    
    def transform(self, X):
        """Transform method."""
        # Define encoding dictionaries
        encoding_dicts = {
            'Attrition_Flag': {'Existing Customer': 0, 'Attrited Customer': 1},
            'Gender': {'M': 1, 'F': 0},
            'Education_Level': {'High School': 1, 'Graduate': 3, 'Uneducated': 0, 'Unknown': -1, 'College': 2, 'Post-Graduate': 4, 'Doctorate': 5},
            'Marital_Status': {'Married': 1, 'Single': 0, 'Unknown': -1, 'Divorced': 2},
            'Income_Category': {'$60K - $80K': 2, 'Less than $40K': 0, '$80K - $120K': 3, '$40K - $60K': 1, '$120K +': 4, 'Unknown': -1},
            'Card_Category': {'Blue': 0, 'Gold': 2, 'Silver': 1, 'Platinum': 3}
        }
        X_encoded = X.copy()
        for column, encoder_dict in encoding_dicts.items():
            X_encoded[column] = X_encoded[column].map(encoder_dict)
        return X_encoded

class StandardizeColumns(BaseEstimator, TransformerMixin):
    """
    Transformer for standardizing numeric columns.

    This class standardizes numeric columns using StandardScaler.

    No parameters are required for initialization.
    """
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        """Fit method."""
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        self.scaler.fit(X[numeric_columns])
        return self
    
    def transform(self, X):
        """Transform method."""
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        X[numeric_columns] = self.scaler.transform(X[numeric_columns])
        return X

class KNNImputeOutliers(BaseEstimator, TransformerMixin):
    """
    Transformer for imputing outliers using KNN.

    This class imputes outliers in numeric columns using KNNImputer.

    No parameters are required for initialization.
    """
    def __init__(self):
        self.imputer = KNNImputer()
    
    def fit(self, X, y=None):
        """Fit method."""
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        self.imputer.fit(X[numeric_columns])
        return self
    
    def transform(self, X):
        """Transform method."""
        numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
        X[numeric_columns] = self.imputer.transform(X[numeric_columns])
        return X

class KNNImputeNegative(BaseEstimator, TransformerMixin):
    """
    Transformer for imputing negative values using KNN.

    This class imputes negative values in object columns using KNNImputer.

    No parameters are required for initialization.
    """
    def __init__(self):
        self.imputer = KNNImputer()
    
    def fit(self, X, y=None):
        """Fit method."""
        return self
    
    def transform(self, X):
        """Transform method."""
        object_columns = X.select_dtypes(include=['object']).columns
        X[object_columns] = self.imputer.fit_transform(X[object_columns])
        X[object_columns] = np.where(X[object_columns] == -1, np.nan, X[object_columns])
        return X

# Define preprocessing steps
preprocessing_steps = [
    ('log_transform', LogTransformSkewedFeatures()),
    ('label_encode', LabelEncodeColumns()),
    ('standardize', StandardizeColumns()),
    ('knn_impute_outliers', KNNImputeOutliers()),
    ('knn_impute_negative', KNNImputeNegative())
]