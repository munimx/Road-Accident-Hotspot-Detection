"""
Task 2: Hotspot Identification using K-Means Clustering
=========================================================
Identify dangerous accident hotspots using K-Means clustering
on GPS coordinates (latitude, longitude).

This identifies geographic clusters of high accident density.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle

def identify_hotspots(filepath, n_clusters=50, output_prefix=None):
    """
    Identify accident hotspots using K-Means clustering
    
    Parameters:
    - filepath: Path to cleaned dataset
    - n_clusters: Number of clusters (default 50)
    - output_prefix: Prefix for saving results
    
    Returns:
    - DataFrame with cluster assignments
    - KMeans model
    - Cluster centers
    """
    print(f"Loading cleaned dataset...")
    df = pd.read_csv(filepath)
    
    # Prepare coordinates for clustering
    coords = df[['Start_Lat', 'Start_Lng']].values
    print(f"Data points for clustering: {coords.shape[0]:,}")
    
    # Standardize coordinates (important for K-Means)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    # Apply K-Means clustering
    print(f"Running K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(coords_scaled)
    
    # Analyze clusters
    print("\nCluster Analysis:")
    cluster_stats = df.groupby('Cluster').agg({
        'ID': 'count',
        'Severity': ['mean', 'max'],
        'Distance(mi)': 'mean',
        'Start_Lat': 'mean',
        'Start_Lng': 'mean'
    }).round(3)
    cluster_stats.columns = ['Count', 'Avg_Severity', 'Max_Severity', 'Avg_Distance', 'Center_Lat', 'Center_Lng']
    cluster_stats = cluster_stats.sort_values('Count', ascending=False)
    
    print("\nTop 10 Hotspots (by accident count):")
    print(cluster_stats.head(10))
    
    # Save results if requested
    if output_prefix:
        df.to_csv(f"{output_prefix}_with_clusters.csv", index=False)
        cluster_stats.to_csv(f"{output_prefix}_hotspot_stats.csv")
        with open(f"{output_prefix}_kmeans_model.pkl", 'wb') as f:
            pickle.dump(kmeans, f)
        print(f"\nResults saved with prefix: {output_prefix}")
    
    return df, kmeans, cluster_stats

if __name__ == "__main__":
    df_clustered, model, stats = identify_hotspots(
        "US_Accidents_Cleaned.csv",
        n_clusters=50,
        output_prefix="hotspots"
    )
