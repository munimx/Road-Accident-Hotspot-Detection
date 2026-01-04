"""
Task 1: Data Cleaning
======================
Clean the large-scale accident dataset:
- Remove/handle missing values
- Validate coordinates
- Remove duplicates
- Standardize data types
- Filter invalid records
"""

import pandas as pd
import numpy as np

def clean_dataset(filepath, output_filepath=None):
    """
    Clean the US Accidents dataset
    
    Parameters:
    - filepath: Input CSV file
    - output_filepath: Where to save cleaned data (optional)
    
    Returns:
    - Cleaned DataFrame
    """
    print("Starting data cleaning...")
    
    # Load dataset
    df = pd.read_csv(filepath)
    print(f"Initial records: {len(df):,}")
    
    # 1. Remove rows with missing critical coordinates
    df = df.dropna(subset=['Start_Lat', 'Start_Lng'])
    print(f"After removing missing coordinates: {len(df):,}")
    
    # 2. Remove duplicate IDs
    df = df.drop_duplicates(subset=['ID'], keep='first')
    print(f"After removing duplicates: {len(df):,}")
    
    # 3. Validate coordinates (USA bounds approximately)
    df = df[(df['Start_Lat'] >= 24) & (df['Start_Lat'] <= 50)]
    df = df[(df['Start_Lng'] >= -125) & (df['Start_Lng'] <= -66)]
    print(f"After validating coordinates: {len(df):,}")
    
    # 4. Remove records with invalid severity
    df = df[df['Severity'].isin([1, 2, 3, 4])]
    print(f"After validating severity: {len(df):,}")
    
    # 5. Convert time columns to datetime
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    df = df.dropna(subset=['Start_Time'])
    print(f"After datetime conversion: {len(df):,}")
    
    # 6. Fill missing values strategically
    df['Distance(mi)'] = df['Distance(mi)'].fillna(0)
    df['City'] = df['City'].fillna('Unknown')
    df['Street'] = df['Street'].fillna('Unknown')
    df['Description'] = df['Description'].fillna('No description')
    
    # 7. Handle boolean columns (ensure no NaN)
    bool_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
                 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    for col in bool_cols:
        df[col] = df[col].fillna(False)
    
    print(f"\nCleaning complete!")
    print(f"Final records: {len(df):,}")
    print(f"Data retained: {(len(df) / 7728394 * 100):.1f}%")
    
    # Save if output path provided
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"Cleaned data saved to: {output_filepath}")
    
    return df

if __name__ == "__main__":
    df_clean = clean_dataset("US_Accidents_March23.csv", 
                            "US_Accidents_Cleaned.csv")
