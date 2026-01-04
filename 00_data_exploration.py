"""
Task 0: Data Exploration & Familiarization
===========================================
This script explores the US Accidents dataset to understand its structure,
data types, missing values, and key statistics.

Dataset: US_Accidents_March23.csv
Records: 7,728,394 accidents
Columns: 46 features including location, severity, weather, time, and road conditions
"""

import pandas as pd
import numpy as np

def explore_dataset(filepath):
    """Load and explore the dataset"""
    print("=" * 80)
    print("US ACCIDENTS DATASET - EXPLORATION")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    
    print(f"\n1. DATASET SHAPE")
    print(f"   Total Records: {df.shape[0]:,}")
    print(f"   Total Columns: {df.shape[1]}")
    
    print(f"\n2. DATA TYPES")
    print(df.dtypes)
    
    print(f"\n3. MISSING VALUES (Top 10)")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing.head(10))
    
    print(f"\n4. KEY STATISTICS")
    print(f"   Severity Levels: {sorted(df['Severity'].unique())}")
    print(f"   States Represented: {df['State'].nunique()}")
    print(f"   Cities: {df['City'].nunique()}")
    print(f"   Date Range: {df['Start_Time'].min()} to {df['Start_Time'].max()}")
    
    print(f"\n5. GEOGRAPHIC COVERAGE")
    print(f"   Latitude Range: {df['Start_Lat'].min():.2f} to {df['Start_Lat'].max():.2f}")
    print(f"   Longitude Range: {df['Start_Lng'].min():.2f} to {df['Start_Lng'].max():.2f}")
    
    print(f"\n6. KEY COLUMNS FOR ANALYSIS")
    print("   - Start_Lat, Start_Lng: Accident coordinates (for clustering)")
    print("   - Severity: 1-4 severity levels")
    print("   - Distance(mi): Impact distance")
    print("   - Weather_Condition: Weather at time of accident")
    print("   - Road Features: Amenity, Bump, Crossing, Junction, Signal, etc.")
    
    return df

if __name__ == "__main__":
    df = explore_dataset("US_Accidents_March23.csv")
