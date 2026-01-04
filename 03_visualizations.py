"""
Task 3: Heatmaps & GPS Coordinate Clustering Visualization
============================================================
Create visualizations of accident hotspots:
- Density heatmaps
- Interactive GPS maps (Folium)
- Cluster scatter plots
- Geographic distribution maps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import warnings
warnings.filterwarnings('ignore')

def create_heatmap_matplotlib(df, output_file=None):
    """Create 2D density heatmap using matplotlib"""
    print("Creating matplotlib heatmap...")
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create 2D histogram heatmap
    h = ax.hist2d(df['Start_Lng'], df['Start_Lat'], 
                   bins=100, cmap='YlOrRd', cmin=1)
    
    plt.colorbar(h[3], ax=ax, label='Accident Density')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('US Accident Density Heatmap (2D Histogram)', fontsize=14, fontweight='bold')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.close()

def create_cluster_scatter(df, output_file=None):
    """Create scatter plot of clusters"""
    print("Creating cluster scatter plot...")
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    scatter = ax.scatter(df['Start_Lng'], df['Start_Lat'], 
                        c=df['Cluster'], cmap='tab20', 
                        alpha=0.6, s=10, edgecolors='none')
    
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('K-Means Clustering of Accidents (GPS Coordinates)', fontsize=14, fontweight='bold')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.close()

def create_interactive_heatmap(df, output_file='accident_heatmap.html'):
    """Create interactive heatmap using Folium"""
    print("Creating interactive Folium heatmap...")
    
    # Sample data if too large (for performance)
    if len(df) > 100000:
        df_sample = df.sample(n=100000, random_state=42)
        print(f"Sampled {len(df_sample):,} records for interactive map")
    else:
        df_sample = df
    
    # Center of US
    center_lat = df['Start_Lat'].mean()
    center_lng = df['Start_Lng'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Add heatmap layer
    heat_data = [[row['Start_Lat'], row['Start_Lng']] 
                 for idx, row in df_sample.iterrows()]
    
    HeatMap(heat_data, radius=15, blur=25, max_zoom=1).add_to(m)
    
    m.save(output_file)
    print(f"Saved: {output_file}")

def create_interactive_clusters(df, output_file='accident_clusters.html'):
    """Create interactive cluster map using Folium"""
    print("Creating interactive Folium cluster map...")
    
    # Sample data if too large
    if len(df) > 50000:
        df_sample = df.sample(n=50000, random_state=42)
        print(f"Sampled {len(df_sample):,} records for interactive map")
    else:
        df_sample = df
    
    center_lat = df['Start_Lat'].mean()
    center_lng = df['Start_Lng'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    # Add cluster markers
    marker_cluster = MarkerCluster().add_to(m)
    
    for idx, row in df_sample.iterrows():
        folium.Marker(
            location=[row['Start_Lat'], row['Start_Lng']],
            popup=f"Severity: {row['Severity']}<br>City: {row['City']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(marker_cluster)
    
    m.save(output_file)
    print(f"Saved: {output_file}")

def create_severity_heatmap(df, output_file=None):
    """Create heatmap colored by severity"""
    print("Creating severity-weighted heatmap...")
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    scatter = ax.scatter(df['Start_Lng'], df['Start_Lat'], 
                        c=df['Severity'], cmap='RdYlGn_r', 
                        alpha=0.5, s=20, edgecolors='none')
    
    plt.colorbar(scatter, ax=ax, label='Severity (1=Low, 4=Critical)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Accident Severity Distribution Across US', fontsize=14, fontweight='bold')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    plt.close()

def create_visualizations(csv_filepath):
    """Create all visualizations"""
    print("Loading data for visualizations...")
    df = pd.read_csv(csv_filepath)
    
    # Sample for faster processing if needed
    if len(df) > 500000:
        print(f"Sampling data from {len(df):,} to 500,000 records for faster processing")
        df = df.sample(n=500000, random_state=42)
    
    print(f"\nCreating visualizations...")
    create_heatmap_matplotlib(df, 'heatmap_density.png')
    create_severity_heatmap(df, 'heatmap_severity.png')
    
    # Only create cluster plot if Cluster column exists
    if 'Cluster' in df.columns:
        create_cluster_scatter(df, 'cluster_scatter.png')
    
    create_interactive_heatmap(df, 'accident_heatmap_interactive.html')
    
    print("\nVisualization creation complete!")

if __name__ == "__main__":
    # Use clustered data if available, otherwise use cleaned data
    try:
        create_visualizations("hotspots_with_clusters.csv")
    except FileNotFoundError:
        print("Clustered data not found, using cleaned data instead...")
        create_visualizations("US_Accidents_Cleaned.csv")
