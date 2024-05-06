import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.cluster import KMeans

# Load the data
churn_data = pd.read_csv('./files/raw_BankChurners.csv')
churn_data = pd.DataFrame(churn_data)
churn_data = churn_data.iloc[:, 1:-2]

# Separate numeric and categorical columns
numeric_columns = churn_data.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = churn_data.select_dtypes(include=['object']).columns

# Transform numeric columns to have a more Gaussian-like distribution
transformer = PowerTransformer(method='yeo-johnson', standardize=False)
churn_data[numeric_columns] = transformer.fit_transform(churn_data[numeric_columns])

# One-hot encode categorical columns and force column names to be string type
encoded_data = pd.get_dummies(churn_data, columns=categorical_columns, drop_first=False)
encoded_data = encoded_data.astype(int)

# Scale all columns in the DataFrame
scaler = MinMaxScaler()
encoded_data_scaled = scaler.fit_transform(encoded_data)

# Convert the transformed data back to a DataFrame
encoded_data_scaled_df = pd.DataFrame(encoded_data_scaled, columns=encoded_data.columns).astype(int)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(encoded_data_scaled_df)
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
clusters = kmeans.fit_predict(encoded_data_scaled_df)

# Separate data for each cluster
cluster_data = {}
for cluster_id in range(5):
    cluster_indices = clusters == cluster_id
    cluster_data[cluster_id] = encoded_data_scaled_df.loc[cluster_indices]

# Perform feature selection using RFE
X_all = encoded_data_scaled_df.drop(columns=['Attrition_Flag_Existing Customer'])
y_all = encoded_data_scaled_df['Attrition_Flag_Existing Customer']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(random_state=42)
rfe = RFE(rf_classifier, n_features_to_select=10)
rfe.fit(X_train, y_train)

selected_features = X_train.columns[rfe.support_]

# Create subplots for confusion matrix and violin plot
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(16, 20))

for cluster_id, data in cluster_data.items():
    # Plot violin plot with scaled data used for clustering
    sns.violinplot(data=encoded_data_scaled_df.loc[data.index, selected_features], orient='h', ax=axes[cluster_id, 0])
    axes[cluster_id, 0].set_title(f'Cluster {cluster_id} - Feature Distribution')
    axes[cluster_id, 0].set_xlabel('Feature Value')
    axes[cluster_id, 0].set_ylabel('Feature Name')

    # Split data for the current cluster
    X_cluster = data[selected_features]
    y_cluster = data['Attrition_Flag_Existing Customer']
    X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
    
    # Train RF classifier and plot confusion matrix
    rf_classifier.fit(X_train_cluster, y_train_cluster)
    y_pred_cluster = rf_classifier.predict(X_test_cluster)
    cm = confusion_matrix(y_test_cluster, y_pred_cluster, labels=[0, 1])  # Specify all known labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=axes[cluster_id, 1])
    axes[cluster_id, 1].set_title(f'Cluster {cluster_id} - Confusion Matrix')
    axes[cluster_id, 1].set_xlabel('Predicted label')
    axes[cluster_id, 1].set_ylabel('True label')

plt.tight_layout()
plt.show()