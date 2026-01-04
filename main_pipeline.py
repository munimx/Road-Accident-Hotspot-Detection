"""
Main Pipeline: Road Accident Hotspot Detection
===============================================
Orchestrates the complete analysis pipeline by executing scripts sequentially
"""

import sys
import os

print("\n" + "="*80)
print("ROAD ACCIDENT HOTSPOT DETECTION - COMPLETE PIPELINE")
print("="*80)

# ============================================================================
# STEP 0: DATA EXPLORATION
# ============================================================================
print("\n[STEP 0] DATA EXPLORATION")
print("-" * 80)
try:
    exec(open("00_data_exploration.py").read())
    print("✓ Data exploration complete")
except Exception as e:
    print(f"✗ Error in exploration: {e}")
    sys.exit(1)

# ============================================================================
# STEP 1: DATA CLEANING
# ============================================================================
print("\n[STEP 1] DATA CLEANING")
print("-" * 80)
try:
    exec(open("01_data_cleaning.py").read())
    print("✓ Data cleaning complete")
except Exception as e:
    print(f"✗ Error in cleaning: {e}")
    sys.exit(1)

# ============================================================================
# STEP 2: HOTSPOT IDENTIFICATION
# ============================================================================
print("\n[STEP 2] HOTSPOT IDENTIFICATION")
print("-" * 80)
try:
    exec(open("02_hotspot_identification.py").read())
    print("✓ Hotspot identification complete")
except Exception as e:
    print(f"✗ Error in hotspot identification: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: VISUALIZATIONS
# ============================================================================
print("\n[STEP 3] VISUALIZATIONS & MAPS")
print("-" * 80)
try:
    exec(open("03_visualizations.py").read())
    print("✓ Visualizations created")
except Exception as e:
    print(f"✗ Error in visualizations: {e}")
    sys.exit(1)

# ============================================================================
# STEP 4: POLICY RECOMMENDATIONS
# ============================================================================
print("\n[STEP 4] POLICY RECOMMENDATIONS")
print("-" * 80)
try:
    exec(open("04_policy_recommendations.py").read())
    print("✓ Policy recommendations generated")
except Exception as e:
    print(f"✗ Error in policy recommendations: {e}")
    sys.exit(1)

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print("\nOutput Files Generated:")
print("  - US_Accidents_Cleaned.csv (cleaned dataset)")
print("  - hotspots_with_clusters.csv (with cluster assignments)")
print("  - hotspot_stats.csv (cluster statistics)")
print("  - heatmap_density.png (density heatmap)")
print("  - heatmap_severity.png (severity heatmap)")
print("  - cluster_scatter.png (cluster visualization)")
print("  - accident_heatmap_interactive.html (interactive map)")
print("\nAnalysis complete. Check outputs for detailed results.")
print("="*80 + "\n")

