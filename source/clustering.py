import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def clustering(churn_data):
    # Apply preprocessing using the preprocess_data function
    clean_data = churn_data.iloc[:, 1:-2]

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

    # Encode 'Attrition_Flag'
    attrition_flag_dict = {'Existing Customer': 0, 'Attrited Customer': 1}
    clean_data['Attrition_Flag'] = clean_data['Attrition_Flag'].map(attrition_flag_dict)

    # One-hot encode the categorical columns
    categorical_columns = clean_data.select_dtypes(include=['object']).columns
    clean_data = pd.get_dummies(clean_data, columns=categorical_columns)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.85)  # Retain 85% of variance
    clean_data_pca = pca.fit_transform(clean_data)

    # Apply feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=5)  # Select 5 best features
    clean_data_selected = selector.fit_transform(clean_data_pca, churn_data['Attrition_Flag'])

    # Determine the optimal number of clusters using the Elbow Method
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(clean_data_selected)
        wcss.append(kmeans.inertia_)

    # Find the optimal number of clusters (the "elbow" point)
    optimal_num_clusters = None
    min_diff = float('inf')
    for i in range(1, len(wcss)):
        diff = wcss[i-1] - wcss[i]
        if diff < min_diff:
            min_diff = diff
            optimal_num_clusters = i + 1

    # Perform KMeans clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
    clusters = kmeans.fit_predict(clean_data_selected)

    # Add 'Cluster_Label' column to original DataFrame
    churn_data['Cluster_Label'] = clusters

    # Save the DataFrame with the 'Cluster_Label' column as a CSV file
    churn_data.to_csv('./files/clustered_BankChurners.csv', index=False)

    return churn_data

# Example usage
churn_data = pd.read_csv('./files/raw_BankChurners.csv')
clusters = clustering(churn_data)