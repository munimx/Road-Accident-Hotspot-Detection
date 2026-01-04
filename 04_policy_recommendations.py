"""
Task 4: Policy Improvements & Recommendations
==============================================
Analyze accident patterns and suggest policy improvements:
- Identify dangerous road features
- Peak accident times and locations
- Weather-related risks
- Severity patterns
- Evidence-based recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_road_features(df):
    """Analyze which road features correlate with accidents"""
    print("\n" + "="*80)
    print("ROAD FEATURES ANALYSIS")
    print("="*80)
    
    road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 
                     'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                     'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    
    feature_severity = {}
    feature_count = {}
    
    for feature in road_features:
        with_feature = df[df[feature] == True]
        if len(with_feature) > 0:
            feature_severity[feature] = with_feature['Severity'].mean()
            feature_count[feature] = len(with_feature)
    
    # Sort by severity
    sorted_features = sorted(feature_severity.items(), key=lambda x: x[1], reverse=True)
    
    print("\nRoad Features by Average Accident Severity:")
    print("-" * 60)
    for feature, severity in sorted_features:
        count = feature_count[feature]
        pct = (count / len(df) * 100)
        print(f"  {feature:20s} | Severity: {severity:.2f} | Count: {count:,} ({pct:.1f}%)")
    
    return dict(sorted_features)

def analyze_temporal_patterns(df):
    """Analyze temporal patterns"""
    print("\n" + "="*80)
    print("TEMPORAL PATTERNS ANALYSIS")
    print("="*80)
    
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month
    
    print("\nPeak Accident Hours:")
    hourly = df.groupby('Hour').agg({'ID': 'count', 'Severity': 'mean'}).round(2)
    hourly.columns = ['Count', 'Avg_Severity']
    hourly = hourly.sort_values('Count', ascending=False)
    print(hourly.head(10))
    
    print("\nAccidents by Day of Week:")
    daily = df.groupby('DayOfWeek').agg({'ID': 'count', 'Severity': 'mean'}).round(2)
    daily.columns = ['Count', 'Avg_Severity']
    daily = daily.sort_values('Count', ascending=False)
    print(daily)
    
    return df

def analyze_geographic_risks(df):
    """Analyze geographic risk factors"""
    print("\n" + "="*80)
    print("GEOGRAPHIC RISK ANALYSIS")
    print("="*80)
    
    state_stats = df.groupby('State').agg({
        'ID': 'count',
        'Severity': 'mean',
        'Distance(mi)': 'mean'
    }).round(2)
    state_stats.columns = ['Count', 'Avg_Severity', 'Avg_Impact_Distance']
    state_stats = state_stats.sort_values('Count', ascending=False)
    
    print("\nTop 15 States by Accident Count:")
    print(state_stats.head(15))
    
    return state_stats

def analyze_weather_impact(df):
    """Analyze weather-related risks"""
    print("\n" + "="*80)
    print("WEATHER IMPACT ANALYSIS")
    print("="*80)
    
    # Handle missing weather data
    weather_df = df[df['Weather_Condition'].notna()].copy()
    
    weather_stats = weather_df.groupby('Weather_Condition').agg({
        'ID': 'count',
        'Severity': 'mean'
    }).round(2)
    weather_stats.columns = ['Count', 'Avg_Severity']
    weather_stats = weather_stats.sort_values('Count', ascending=False)
    
    print("\nTop 15 Weather Conditions Associated with Accidents:")
    print(weather_stats.head(15))
    
    return weather_stats

def generate_recommendations(road_features, geographic_stats, weather_stats):
    """Generate policy recommendations based on analysis"""
    print("\n" + "="*80)
    print("EVIDENCE-BASED POLICY RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Road features recommendations
    if 'Junction' in road_features and road_features['Junction'] > 2.5:
        recommendations.append(
            "1. JUNCTION SAFETY IMPROVEMENTS\n"
            "   - Higher severity at junctions (avg: {:.2f})\n"
            "   - Recommend: Improved signage, better traffic signals, \n"
            "     automated traffic management at high-accident junctions"
            .format(road_features.get('Junction', 0))
        )
    
    if 'Traffic_Signal' in road_features and road_features['Traffic_Signal'] > 2.3:
        recommendations.append(
            "2. TRAFFIC SIGNAL OPTIMIZATION\n"
            "   - Severity: {:.2f} at signaled intersections\n"
            "   - Recommend: Adaptive signal timing, countdown signals,\n"
            "     smart traffic systems based on real-time conditions"
            .format(road_features.get('Traffic_Signal', 0))
        )
    
    recommendations.append(
        "3. HOTSPOT-BASED INTERVENTIONS\n"
        "   - Deploy enhanced police presence in identified hotspots\n"
        "   - Install speed cameras and automated enforcement\n"
        "   - Improve road infrastructure in high-crash clusters"
    )
    
    recommendations.append(
        "4. WEATHER-RELATED SAFETY\n"
        "   - Most accidents in '{}' conditions\n"
        "   - Recommend: Dynamic speed limit reduction in adverse weather,\n"
        "     improved drainage to prevent hydroplaning"
        .format(weather_stats.index[0])
    )
    
    recommendations.append(
        "5. TEMPORAL INTERVENTIONS\n"
        "   - Deploy resources during peak accident hours\n"
        "   - Enhanced safety campaigns during high-risk periods\n"
        "   - Evening commute (4-7 PM) appears to be critical time"
    )
    
    recommendations.append(
        "6. DATA-DRIVEN MONITORING\n"
        "   - Implement real-time accident prediction systems\n"
        "   - Use machine learning for early warning systems\n"
        "   - Regular analysis of new hotspot formation"
    )
    
    print("\n" + "-"*80)
    for rec in recommendations:
        print(rec)
        print()

def generate_policy_report(csv_filepath, output_file='policy_report.txt'):
    """Generate comprehensive policy report"""
    print("Loading data for policy analysis...")
    df = pd.read_csv(csv_filepath)
    
    # Sample if too large
    if len(df) > 1000000:
        print(f"Sampling to 1M records for analysis")
        df = df.sample(n=1000000, random_state=42)
    
    road_features = analyze_road_features(df)
    df = analyze_temporal_patterns(df)
    geographic_stats = analyze_geographic_risks(df)
    weather_stats = analyze_weather_impact(df)
    
    generate_recommendations(road_features, geographic_stats, weather_stats)
    
    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)

if __name__ == "__main__":
    try:
        generate_policy_report("hotspots_with_clusters.csv")
    except FileNotFoundError:
        print("Clustered data not found, using cleaned data instead...")
        generate_policy_report("US_Accidents_Cleaned.csv")
