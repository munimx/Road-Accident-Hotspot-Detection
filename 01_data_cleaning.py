"""
Task 1: Data Cleaning for K-Means Clustering
=============================================
Clean the US Accidents dataset (7.7M records, 47 columns) for two-stage analysis:
  Stage 1: Geographic hotspot identification using K-Means clustering on GPS coordinates
  Stage 2: Post-clustering policy analysis using categorical and numerical features

Steps:
1. Load data with feature selection (30 columns from 47)
2. Remove missing critical values
3. Remove duplicates
4. Validate coordinate bounds (USA continental)
5. Remove coordinate outliers (z-score method)
6. Validate severity values
7. Process datetime columns
8. Handle missing values (numerical, categorical, boolean)
9. Feature engineering (temporal features, duration)
10. Final quality checks
11. Save cleaned dataset with comprehensive report
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def clean_dataset(filepath, output_filepath=None):
    """
    Clean the US Accidents dataset for K-Means clustering analysis.
    
    Parameters:
    - filepath: Input CSV file path
    - output_filepath: Where to save cleaned data (optional)
    
    Returns:
    - Cleaned DataFrame ready for clustering
    """
    
    # Track statistics for final report
    cleaning_stats = {}
    
    # =========================================================================
    # STEP 1: Load Data with Feature Selection
    # =========================================================================
    print("=" * 80)
    print("STEP 1: Loading data with feature selection...")
    print("=" * 80)
    
    columns_to_keep = [
        # Clustering features
        'Start_Lat', 'Start_Lng',
        # Identifiers & Impact
        'ID', 'Severity', 'Start_Time', 'End_Time', 'Distance(mi)',
        # Geographic (for groupby analysis)
        'City', 'State', 'County',
        # Weather features
        'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
        'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 'Wind_Direction',
        # Road features (boolean POI annotations)
        'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
        'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
        'Traffic_Signal', 'Turning_Loop',
        # Time of day
        'Sunrise_Sunset'
    ]
    
    df = pd.read_csv(filepath, usecols=columns_to_keep)
    initial_records = len(df)
    cleaning_stats['initial_records'] = initial_records
    print(f"  Loaded {initial_records:,} records with {len(columns_to_keep)} columns")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # =========================================================================
    # STEP 2: Remove Missing Critical Values
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Removing missing critical values...")
    print("=" * 80)
    
    critical_cols = ['Start_Lat', 'Start_Lng', 'Severity', 'Start_Time']
    before = len(df)
    
    # Log missing counts before removal
    for col in critical_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"  {col}: {missing:,} missing values")
    
    df = df.dropna(subset=critical_cols)
    after = len(df)
    cleaning_stats['after_critical_nulls'] = after
    print(f"  Records removed: {before - after:,}")
    print(f"  Records remaining: {after:,}")
    
    # =========================================================================
    # STEP 3: Remove Duplicates
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Removing duplicate IDs...")
    print("=" * 80)
    
    before = len(df)
    duplicates = df['ID'].duplicated().sum()
    df = df.drop_duplicates(subset=['ID'], keep='first')
    after = len(df)
    cleaning_stats['after_duplicates'] = after
    print(f"  Duplicate IDs found: {duplicates:,}")
    print(f"  Records remaining: {after:,}")
    
    # =========================================================================
    # STEP 4: Validate Coordinate Bounds
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Validating coordinate bounds (USA continental)...")
    print("=" * 80)
    print("  Bounds: Latitude 24°N-50°N, Longitude -125°W to -66°W")
    
    before = len(df)
    
    # Check out-of-bounds before filtering
    lat_invalid = ((df['Start_Lat'] < 24) | (df['Start_Lat'] > 50)).sum()
    lng_invalid = ((df['Start_Lng'] < -125) | (df['Start_Lng'] > -66)).sum()
    print(f"  Latitude out of bounds: {lat_invalid:,}")
    print(f"  Longitude out of bounds: {lng_invalid:,}")
    
    df = df[(df['Start_Lat'] >= 24) & (df['Start_Lat'] <= 50)]
    df = df[(df['Start_Lng'] >= -125) & (df['Start_Lng'] <= -66)]
    after = len(df)
    cleaning_stats['after_coord_bounds'] = after
    print(f"  Records removed: {before - after:,}")
    print(f"  Records remaining: {after:,}")
    
    # =========================================================================
    # STEP 5: Remove Coordinate Outliers (Z-Score Method)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Removing coordinate outliers (z-score > 3)...")
    print("=" * 80)
    
    before = len(df)
    
    # Calculate z-scores for coordinates
    z_scores = np.abs(stats.zscore(df[['Start_Lat', 'Start_Lng']]))
    outlier_mask = (z_scores < 3).all(axis=1)
    outliers_count = (~outlier_mask).sum()
    
    df = df[outlier_mask]
    after = len(df)
    cleaning_stats['after_outliers'] = after
    print(f"  Outliers detected: {outliers_count:,}")
    print(f"  Records remaining: {after:,}")
    
    # =========================================================================
    # STEP 6: Validate Severity
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Validating severity values (1-4)...")
    print("=" * 80)
    
    before = len(df)
    invalid_severity = (~df['Severity'].isin([1, 2, 3, 4])).sum()
    df = df[df['Severity'].isin([1, 2, 3, 4])]
    after = len(df)
    cleaning_stats['after_severity'] = after
    print(f"  Invalid severity values: {invalid_severity:,}")
    print(f"  Records remaining: {after:,}")
    
    # =========================================================================
    # STEP 7: Process Datetime Columns
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Processing datetime columns...")
    print("=" * 80)
    
    before = len(df)
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    invalid_datetime = df['Start_Time'].isnull().sum()
    df = df.dropna(subset=['Start_Time'])
    after = len(df)
    cleaning_stats['after_datetime'] = after
    print(f"  Invalid Start_Time values: {invalid_datetime:,}")
    print(f"  Records remaining: {after:,}")
    
    # =========================================================================
    # STEP 8: Handle Missing Values - Numerical Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: Handling missing numerical values (median imputation)...")
    print("=" * 80)
    
    # Distance
    distance_nulls = df['Distance(mi)'].isnull().sum()
    if distance_nulls > 0:
        median_val = df['Distance(mi)'].median()
        df['Distance(mi)'] = df['Distance(mi)'].fillna(median_val)
        print(f"  Distance(mi): {distance_nulls:,} nulls filled with median ({median_val:.2f})")
    else:
        print(f"  Distance(mi): No missing values")
    
    # Weather numerical features
    weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    
    for col in weather_cols:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  {col}: {nulls:,} nulls filled with median ({median_val:.2f})")
            else:
                print(f"  {col}: No missing values")
    
    # =========================================================================
    # STEP 9: Handle Missing Values - Categorical Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: Handling missing categorical values...")
    print("=" * 80)
    
    # Weather categorical
    weather_cond_nulls = df['Weather_Condition'].isnull().sum()
    df['Weather_Condition'] = df['Weather_Condition'].fillna('Clear')
    print(f"  Weather_Condition: {weather_cond_nulls:,} nulls filled with 'Clear'")
    
    wind_dir_nulls = df['Wind_Direction'].isnull().sum()
    df['Wind_Direction'] = df['Wind_Direction'].fillna('CALM')
    print(f"  Wind_Direction: {wind_dir_nulls:,} nulls filled with 'CALM'")
    
    # Location categorical
    city_nulls = df['City'].isnull().sum()
    df['City'] = df['City'].fillna('Unknown')
    print(f"  City: {city_nulls:,} nulls filled with 'Unknown'")
    
    state_nulls = df['State'].isnull().sum()
    df['State'] = df['State'].fillna('Unknown')
    print(f"  State: {state_nulls:,} nulls filled with 'Unknown'")
    
    county_nulls = df['County'].isnull().sum()
    df['County'] = df['County'].fillna('Unknown')
    print(f"  County: {county_nulls:,} nulls filled with 'Unknown'")
    
    # Time of day
    sunrise_nulls = df['Sunrise_Sunset'].isnull().sum()
    df['Sunrise_Sunset'] = df['Sunrise_Sunset'].fillna('Unknown')
    print(f"  Sunrise_Sunset: {sunrise_nulls:,} nulls filled with 'Unknown'")
    
    # =========================================================================
    # STEP 10: Handle Boolean Road Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 10: Handling boolean road features (fill with False)...")
    print("=" * 80)
    
    bool_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
                 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    
    total_bool_nulls = 0
    for col in bool_cols:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            if nulls > 0:
                df[col] = df[col].fillna(False)
                total_bool_nulls += nulls
                print(f"  {col}: {nulls:,} nulls filled with False")
    
    if total_bool_nulls == 0:
        print("  All boolean columns complete - no missing values")
    else:
        print(f"  Total boolean nulls filled: {total_bool_nulls:,}")
    
    # =========================================================================
    # STEP 11: Feature Engineering - Temporal Features
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 11: Creating temporal features...")
    print("=" * 80)
    
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['Month'] = df['Start_Time'].dt.month
    df['Year'] = df['Start_Time'].dt.year
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    print("  Created features:")
    print("    - Hour (0-23)")
    print("    - DayOfWeek (0=Monday, 6=Sunday)")
    print("    - Month (1-12)")
    print("    - Year")
    print("    - IsWeekend (0=Weekday, 1=Weekend)")
    
    # =========================================================================
    # STEP 12: Feature Engineering - Duration
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 12: Calculating accident duration...")
    print("=" * 80)
    
    df['Duration_hours'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 3600
    
    # Handle invalid durations
    invalid_duration = ((df['Duration_hours'] < 0) | (df['Duration_hours'] > 24) | 
                        df['Duration_hours'].isnull()).sum()
    df['Duration_hours'] = df['Duration_hours'].clip(lower=0, upper=24)
    df['Duration_hours'] = df['Duration_hours'].fillna(df['Duration_hours'].median())
    
    print(f"  Duration_hours created (clipped to 0-24 hours)")
    print(f"  Invalid/extreme durations adjusted: {invalid_duration:,}")
    print(f"  Median duration: {df['Duration_hours'].median():.2f} hours")
    
    # =========================================================================
    # STEP 13: Final Quality Checks
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 13: Running final quality checks...")
    print("=" * 80)
    
    checks_passed = True
    
    # Check 1: No nulls in critical columns
    critical_cols = ['Start_Lat', 'Start_Lng', 'Severity', 'Start_Time', 'ID']
    for col in critical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            print(f"  ✗ FAIL: {col} has {null_count:,} null values")
            checks_passed = False
        else:
            print(f"  ✓ PASS: {col} - no null values")
    
    # Check 2: Coordinate ranges valid
    lat_valid = df['Start_Lat'].between(24, 50).all()
    lng_valid = df['Start_Lng'].between(-125, -66).all()
    if lat_valid and lng_valid:
        print(f"  ✓ PASS: All coordinates within valid bounds")
    else:
        print(f"  ✗ FAIL: Some coordinates out of bounds")
        checks_passed = False
    
    # Check 3: Severity valid
    severity_valid = df['Severity'].isin([1, 2, 3, 4]).all()
    if severity_valid:
        print(f"  ✓ PASS: All severity values valid (1-4)")
    else:
        print(f"  ✗ FAIL: Invalid severity values found")
        checks_passed = False
    
    # Check 4: No duplicates
    duplicate_count = df['ID'].duplicated().sum()
    if duplicate_count == 0:
        print(f"  ✓ PASS: No duplicate IDs")
    else:
        print(f"  ✗ FAIL: {duplicate_count:,} duplicate IDs found")
        checks_passed = False
    
    if checks_passed:
        print("\n  ★ ALL QUALITY CHECKS PASSED ★")
    else:
        print("\n  ⚠ SOME QUALITY CHECKS FAILED - Review data")
    
    # =========================================================================
    # STEP 14: Save Cleaned Dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 14: Saving cleaned dataset...")
    print("=" * 80)
    
    final_records = len(df)
    cleaning_stats['final_records'] = final_records
    
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"  Saved to: {output_filepath}")
        
        # Save column reference
        columns_file = output_filepath.replace('.csv', '_columns.txt')
        with open(columns_file, 'w') as f:
            f.write("US Accidents Cleaned Dataset - Column Reference\n")
            f.write("=" * 50 + "\n\n")
            for i, col in enumerate(df.columns, 1):
                f.write(f"{i:2d}. {col}\n")
        print(f"  Column reference saved to: {columns_file}")
    
    # =========================================================================
    # COMPREHENSIVE SUMMARY REPORT
    # =========================================================================
    print("\n")
    print("=" * 80)
    print("DATA CLEANING SUMMARY")
    print("=" * 80)
    
    records_removed = initial_records - final_records
    retention_rate = (final_records / initial_records) * 100
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    print(f"Initial records:      {initial_records:,}")
    print(f"Final records:        {final_records:,}")
    print(f"Records removed:      {records_removed:,}")
    print(f"Data retention rate:  {retention_rate:.2f}%")
    print(f"\nFinal columns:        {len(df.columns)}")
    print(f"Memory usage:         {memory_mb:.2f} MB")
    
    print("\n" + "=" * 80)
    print("FEATURE BREAKDOWN")
    print("=" * 80)
    
    print("Clustering Features (2):")
    print("  - Start_Lat, Start_Lng")
    
    print("\nTemporal Features (6):")
    print("  - Hour, DayOfWeek, Month, Year, IsWeekend, Duration_hours")
    
    print("\nWeather Features (8):")
    print("  - Temperature(F), Humidity(%), Pressure(in), Visibility(mi)")
    print("  - Wind_Speed(mph), Precipitation(in), Weather_Condition, Wind_Direction")
    
    print("\nRoad Features (13 boolean):")
    print("  - Amenity, Bump, Crossing, Give_Way, Junction, No_Exit, Railway")
    print("  - Roundabout, Station, Stop, Traffic_Calming, Traffic_Signal, Turning_Loop")
    
    print("\nLocation Features (3):")
    print("  - City, State, County")
    
    print("\nOther Analysis Features (4):")
    print("  - ID, Severity, Distance(mi), Sunrise_Sunset")
    
    print("\n" + "=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)
    
    print("Coordinate Statistics:")
    print(f"  Latitude range:  {df['Start_Lat'].min():.4f} to {df['Start_Lat'].max():.4f}")
    print(f"  Longitude range: {df['Start_Lng'].min():.4f} to {df['Start_Lng'].max():.4f}")
    
    print("\nSeverity Distribution:")
    for sev in [1, 2, 3, 4]:
        count = (df['Severity'] == sev).sum()
        pct = count / final_records * 100
        print(f"  Severity {sev}: {count:,} ({pct:.1f}%)")
    
    print("\nTemporal Coverage:")
    date_min = df['Start_Time'].min().strftime('%Y-%m-%d')
    date_max = df['Start_Time'].max().strftime('%Y-%m-%d')
    total_days = (df['Start_Time'].max() - df['Start_Time'].min()).days
    print(f"  Date range: {date_min} to {date_max}")
    print(f"  Total days: {total_days:,}")
    
    print("\nTop 5 States:")
    top_states = df['State'].value_counts().head(5)
    for state, count in top_states.items():
        print(f"  {state}: {count:,}")
    
    print("\nNull Value Check:")
    print("  ✓ No null values in critical columns")
    print("  ✓ All coordinates within valid bounds")
    print("  ✓ All severity values valid (1-4)")
    print("  ✓ No duplicate IDs")
    
    # Check remaining nulls
    remaining_nulls = df.isnull().sum()
    remaining_nulls = remaining_nulls[remaining_nulls > 0]
    if len(remaining_nulls) > 0:
        print("\nRemaining nulls (non-critical):")
        for col, count in remaining_nulls.items():
            print(f"  {col}: {count:,}")
    else:
        print("\nRemaining nulls (non-critical):")
        print("  None - all columns complete")
    
    print("\n" + "=" * 80)
    print("FILES CREATED")
    print("=" * 80)
    if output_filepath:
        print(f"1. {output_filepath} - Cleaned dataset")
        print(f"2. {output_filepath.replace('.csv', '_columns.txt')} - Column reference list")
    
    print("\n" + "=" * 80)
    print("READY FOR K-MEANS CLUSTERING")
    print("=" * 80)
    print("✓ Clustering features: Start_Lat, Start_Lng (ready for StandardScaler)")
    print("✓ Analysis features: All categorical and numerical features preserved")
    print("✓ No missing values in critical columns")
    print("✓ All validation rules passed")
    print("\nNext Step: Run 02_hotspot_identification.py")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    df_clean = clean_dataset(
        "US_Accidents_March23.csv", 
        "US_Accidents_Cleaned.csv"
    )
