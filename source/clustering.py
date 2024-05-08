import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.cluster import KMeans
from utils import preprocess_data

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

    # Determine the optimal number of clusters using the Elbow Method
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(clean_data)
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
    clusters = kmeans.fit_predict(clean_data)
       
    # Get count of data points in each cluster
    cluster_counts = pd.Series(clusters).value_counts().sort_index()

    # Count occurrences of each target variable within each cluster
    cluster_targets = pd.concat([pd.Series(clusters, name='Cluster'), churn_data['Attrition_Flag']], axis=1)
    cluster_targets['Count'] = 1
    cluster_targets = cluster_targets.groupby(['Cluster', 'Attrition_Flag']).count().reset_index()
    
    # Print cluster details
    print("Number of clusters:", optimal_num_clusters)
    for i in range(optimal_num_clusters):
        print(f"Cluster {i + 1}")
        print(f"Number of data points in Cluster {i + 1}:", cluster_counts[i])
        print(f"Target variable distribution in Cluster {i + 1}:")
        total_points = cluster_counts[i]
        for _, row in cluster_targets[cluster_targets['Cluster'] == i].iterrows():
            target_value = row['Attrition_Flag']
            count = row['Count']
            percentage = (count / total_points) * 100
            print(f"   Target: {target_value}, Percentage: {percentage:.2f}%")

    return clusters

# Example usage
churn_data = pd.read_csv('./files/raw_BankChurners.csv')
clusters = clustering(churn_data)