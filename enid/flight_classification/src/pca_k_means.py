# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from sklearn.metrics import silhouette_score

# def extract_trajectory_features(df):
#     """Extracts meaningful features from flight data."""
#     features = []
    
#     for flight_id, flight in df.groupby("flight_id"):
#         lat_var = flight["lat"].var()
#         long_var = flight["long"].var()
#         alt_var = flight["altitude"].var()
#         speed_var = flight["speed"].var()
#         avg_speed = flight["speed"].mean()
#         avg_altitude = flight["altitude"].mean()
#         max_altitude = flight["altitude"].max()
#         min_altitude = flight["altitude"].min()
#         altitude_range = max_altitude - min_altitude

#         # Compute heading changes
#         flight = flight.copy()
#         flight["heading"] = np.arctan2(flight["long"].diff(), flight["lat"].diff())
#         flight["heading_change"] = flight["heading"].diff().abs()

#         # Count alternating direction changes (grid pattern)
#         alternating_changes = ((flight["heading_change"] > 0.2) & (flight["heading_change"].shift() > 0.2)).sum()

#         features.append([flight_id, lat_var, long_var, alt_var, speed_var, avg_speed, avg_altitude, 
#                          altitude_range, alternating_changes])

#     return pd.DataFrame(features, columns=[
#         "flight_id", "lat_var", "long_var", "alt_var", "speed_var", 
#         "avg_speed", "avg_altitude", "altitude_range", "alternating_changes"
#     ])

# def analyze_flight_data_clustering(df, n_clusters=5):
#     """Applies K-Means clustering to categorize flights and computes Silhouette Score and Inertia."""
#     feature_df = extract_trajectory_features(df)

#     # Standardize features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(feature_df.iloc[:, 1:])

#     # Apply PCA for dimensionality reduction (optional, but helps with visualization)
#     pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
#     X_pca = pca.fit_transform(X_scaled)

#     # Plot the PCA components to inspect how well the clusters are separated
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', label="Flight Paths")
#     plt.title("PCA of Flight Path Features")
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.show()

#     # Apply K-Means
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)

#     # Calculate Silhouette Score
#     silhouette_avg = silhouette_score(X_scaled, labels)
#     print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.4f}")

#     # Calculate Inertia (Confidence Metric)
#     inertia = kmeans.inertia_
#     print(f"Inertia (Confidence Metric) for {n_clusters} clusters: {inertia:.4f}")

#     # Visualizing the cluster results with PCA reduced dimensions
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
#     plt.title(f"Clusters (K={n_clusters}), Silhouette Score: {silhouette_avg:.2f}")
#     plt.xlabel("PCA Component 1")
#     plt.ylabel("PCA Component 2")
#     plt.show()

#     # Assign categories based on new cluster centers
#     cluster_mapping = create_cluster_mapping(kmeans, scaler, feature_df.columns[1:])

#     return [cluster_mapping[label] for label in labels], silhouette_avg, inertia

def create_cluster_mapping(kmeans, scaler, feature_names):
    """Refines cluster labeling based on updated feature understanding."""
    centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Convert back to original scale
    df_centers = pd.DataFrame(centers, columns=feature_names)

    print("Cluster Centers:\n", df_centers)  # Debugging step

    mapping = {}

    for i, row in df_centers.iterrows():
        print(f"Cluster {i}: {row.to_dict()}")  # Inspect each cluster's features

        if row["avg_altitude"] > 10000 and row["avg_speed"] > 200:
            mapping[i] = "Likely Commercial"
        elif row["avg_speed"] < 150 and row["altitude_range"] < 4000 and row["alternating_changes"] > 50:
            mapping[i] = "Likely Surveillance"
        elif row["avg_altitude"] < 10000 and row["altitude_range"] > 5000 and row["alternating_changes"] > 50:
            mapping[i] = "Likely Emergency"
        elif row["avg_speed"] > 150 and row["altitude_range"] > 3000 and row["alternating_changes"] < 10:
            mapping[i] = "Likely Private"
        elif row["avg_speed"] < 200 and row["altitude_range"] < 5000 and row["alternating_changes"] > 100:
            mapping[i] = "Likely Training"
        else:
            mapping[i] = "Likely Training"


    return mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt


def analyze_flight_data_clustering(df, n_clusters=5):
    """Applies K-Means clustering to categorize flights and computes Silhouette Score and Inertia."""
    feature_df = extract_trajectory_features(df)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.iloc[:, 1:])

    # Apply PCA for dimensionality reduction (optional, but helps with visualization)
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    X_pca = pca.fit_transform(X_scaled)

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Step 4: Compute silhouette scores
    silhouette_avg = silhouette_score(X_scaled, labels)  # Single overall score
    per_sample_silhouettes = silhouette_samples(X_scaled, labels)  # Per-point scores

    
    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.4f}")

    # Calculate Inertia (Confidence Metric)
    inertia = kmeans.inertia_
    print(f"Inertia (Confidence Metric) for {n_clusters} clusters: {inertia:.4f}")

    # Visualizing the cluster results with PCA reduced dimensions
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f"Clusters (K={n_clusters}), Silhouette Score: {silhouette_avg:.2f}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    cluster_mapping = create_cluster_mapping(kmeans, scaler, feature_df.columns[1:])
    print("Cluster Mapping:", cluster_mapping)
    # Map cluster labels back to the original data by flight_id
    flight_ids = df['flight_id'].unique()

    # Ensure the cluster labels match flight_ids
    flight_to_label = dict(zip(flight_ids, labels))  # Create a mapping from flight_id to cluster label
    
    flight_to_silhouette = dict(zip(flight_ids, per_sample_silhouettes))  # Mapping from flight_id to silhouette score
    df["silhouette_score"] = df["flight_id"].map(flight_to_silhouette)  # Add silhouette score to DataFrame

    # Create a new column 'cluster_label' that maps the flight_id to the cluster label
    df['cluster_label'] = df['flight_id'].map(flight_to_label)
    df['label'] = df['cluster_label'].map(cluster_mapping)


        # Compute inertia per cluster
    cluster_inertia = {label: df[df['cluster_label'] == label]['silhouette_score'].var() for label in df['cluster_label'].unique()}

    # Convert inertia to confidence scores
    df['confidence_score'] = df['cluster_label'].map(lambda label: 1 / (1 + cluster_inertia[label]))

    # Display confidence scores
    print(df.head())


    # After clustering, generate the classification report
    generate_flight_classification_report(df, silhouette_avg, inertia)

    return df


def extract_trajectory_features(df):
    """Extracts meaningful features from flight data."""
    features = []
    
    for flight_id, flight in df.groupby("flight_id"):
        lat_var = flight["lat"].var()
        long_var = flight["long"].var()
        alt_var = flight["altitude"].var()
        speed_var = flight["speed"].var()
        avg_speed = flight["speed"].mean()
        avg_altitude = flight["altitude"].mean()
        max_altitude = flight["altitude"].max()
        min_altitude = flight["altitude"].min()
        altitude_range = max_altitude - min_altitude

        # Compute heading changes
        flight = flight.copy()
        flight["heading"] = np.arctan2(flight["long"].diff(), flight["lat"].diff())
        flight["heading_change"] = flight["heading"].diff().abs()

        # Count alternating direction changes (grid pattern)
        alternating_changes = ((flight["heading_change"] > 0.2) & (flight["heading_change"].shift() > 0.2)).sum()

        features.append([flight_id, lat_var, long_var, alt_var, speed_var, avg_speed, avg_altitude, 
                         altitude_range, alternating_changes])

    return pd.DataFrame(features, columns=[ 
        "flight_id", "lat_var", "long_var", "alt_var", "speed_var", 
        "avg_speed", "avg_altitude", "altitude_range", "alternating_changes"
    ])


def generate_flight_classification_report(df, silhouette_avg, inertia, report_filename='flight_classification_report.csv'):
    """
    Generates a report for the flight classification results.
    
    Parameters:
    - df (DataFrame): DataFrame containing flight data with a 'silhouette_score' column.
    - labels (array): Cluster labels for each flight.
    - silhouette_avg (float): The overall silhouette score.
    - inertia (float): The inertia score.
    - report_filename (str): The filename to save the summary report as a CSV file.
    

    Returns:
    - report_df (DataFrame): Summary statistics for the report.
    """
    

    # 1. Summary of Flight Labels
    label_counts = df.groupby('flight_id')['label'].first().value_counts()
    print("Flight Label Counts:")
    print(label_counts)

    # 2. Silhouette Score Statistics
    silhouette_stats = df['silhouette_score'].describe()
    print("\nSilhouette Score Statistics:")
    print(silhouette_stats)

    # 2. Confidence Score Distribution
    confidence_stats = df['confidence_score'].describe()
    print("\nConfidence Score Statistics:")
    print(confidence_stats)

    # 3. Visualize Silhouette Score Distribution
    unique_flight_scores = df.groupby('flight_id')['silhouette_score'].first()
    plt.figure(figsize=(10, 6))
    plt.hist(unique_flight_scores, bins=20, edgecolor='k', color='lightgreen')
    plt.title('Silhouette Score Distribution')
    plt.xlabel('Silhouette Score')
    plt.ylabel('Frequency')
    plt.show()

    unique_confidence_scores = df.groupby('flight_id')['confidence_score'].first()
    plt.figure(figsize=(10, 6))
    plt.hist(unique_confidence_scores, bins=20, edgecolor='k', color='skyblue')
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.show()

    # 4. Flight Label Distribution (Bar Chart)
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar', color='lightcoral')
    plt.title('Flight Type Distribution')
    plt.xlabel('Flight Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

    # 5. Clustering Metrics
    print("\nClustering Metrics:")
    print(f"Overall Silhouette Score: {silhouette_avg}")
    print(f"Inertia: {inertia}")

    # 6. Generate summary report
    report = {
        'Confidence Score Statistics': confidence_stats,
        'Silhouette Score Statistics': silhouette_stats,
    }

    # Create a DataFrame for the report summary
    report_df = pd.DataFrame(report)
    print("\nFlight Classification Report:")
    print(report_df)
    # # Save the report as a CSV
    # report_df.to_csv(report_filename, index=False)
    # print(f"\nReport saved as {report_filename}")

    # return report_df

