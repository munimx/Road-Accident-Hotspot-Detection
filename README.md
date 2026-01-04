# Road Accident Hotspot Detection

Identify dangerous road accident hotspots across the US using machine learning
clustering.

## What This Project Does

This project finds geographic areas where car accidents frequently happen. It
takes 7.7 million accident records, groups them into clusters using K-Means
algorithm, and creates maps showing where accidents are most concentrated. It
also analyzes patterns to suggest policy improvements.

## The Dataset

**US Accidents (March 2023)**

- **7.7 million** accident records from all 50 US states
- **Key information:** GPS location, severity (1-4), weather, time, road
  features
- **File:** `US_Accidents_March23.csv` (3GB)

The data includes when and where each accident happened, how severe it was, and
what conditions existed (weather, road type, etc.).

## How It Works - 5 Steps

### 1. Explore Data

**File:** `00_data_exploration.py`

Understand what we're working with:

- How many records? (7.7 million)
- What info do we have? (location, severity, weather, etc.)
- Any missing data?

### 2. Clean Data

**File:** `01_data_cleaning.py`

Remove bad data so we can analyze:

- Remove records with invalid locations
- Remove duplicates
- Remove incomplete records
- Output: Clean dataset (~7.3 million records)

### 3. Find Hotspots

**File:** `02_hotspot_identification.py`

Group accidents into clusters:

- Uses K-Means machine learning algorithm
- Creates 50 geographic clusters
- Finds which areas have the most accidents
- Output: Data with cluster assignments + statistics

### 4. Create Maps

**File:** `03_visualizations.py`

Make visual representations:

- **Density heatmap:** Shows where accidents concentrate
- **Severity map:** Shows which areas are most dangerous
- **Cluster map:** Shows the 50 identified hotspots
- **Interactive map:** Explore in your browser (HTML file)
- Output: PNG images + interactive HTML map

### 5. Recommend Policies

**File:** `04_policy_recommendations.py`

Analyze the data and suggest improvements:

- Which road features cause more accidents?
- What time of day is most dangerous?
- Which states have highest accident rates?
- How does weather impact accidents?
- Suggests evidence-based safety policies

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run everything
python main_pipeline.py
```

The pipeline runs all 5 steps and generates outputs in about **7-8 minutes**.

## Output Files

| File                                | What It Is                          |
| ----------------------------------- | ----------------------------------- |
| `US_Accidents_Cleaned.csv`          | Clean dataset ready for analysis    |
| `hotspots_with_clusters.csv`        | Original data + cluster assignments |
| `hotspot_stats.csv`                 | Statistics for each cluster         |
| `heatmap_density.png`               | Map showing accident density        |
| `heatmap_severity.png`              | Map colored by accident severity    |
| `cluster_scatter.png`               | Map showing the 50 clusters         |
| `accident_heatmap_interactive.html` | Interactive map (open in browser)   |

## Run Individual Steps

Instead of running everything at once, you can run each step separately:

```bash
python 00_data_exploration.py        # Understand data
python 01_data_cleaning.py           # Clean data
python 02_hotspot_identification.py  # Find hotspots
python 03_visualizations.py          # Create maps
python 04_policy_recommendations.py  # Analyze & recommend
```

## Key Technologies

- **pandas:** Load and process the large CSV file
- **scikit-learn:** K-Means clustering algorithm
- **matplotlib & seaborn:** Create charts and heatmaps
- **folium:** Create interactive maps

## What You'll Learn

- How to work with large datasets (7.7 million records)
- Data cleaning and validation
- Machine learning clustering algorithms
- Data visualization techniques
- How to turn analysis into actionable insights

## Example Results

After running the pipeline, you get:

- **Hotspot locations:** Geographic clusters of accidents
- **Risk analysis:** Which areas are most dangerous
- **Temporal patterns:** When accidents happen most
- **Weather impact:** How conditions affect accidents
- **Policy recommendations:** Evidence-based safety improvements

## Policy Recommendations

1. **Junction Safety:** Deploy enhanced enforcement at cluster centers
2. **Smart Traffic:** Implement adaptive signal timing in hotspots
3. **Hotspot Response:** Pre-position emergency services in identified clusters
4. **Weather Adaptation:** Dynamic speed limits during adverse conditions
5. **Temporal Strategy:** Increase patrols during peak accident hours
6. **Predictive Systems:** Use ML for early warning and prevention

## Future Ideas

- Predict accident risk in real-time
- Seasonal trend analysis
- Integration with live traffic data
- Cause prediction (what causes accidents in each hotspot)
- Comparison with traffic volume patterns

## Acknowledgements

**Dataset Source:**
[US Accidents (2016 - 2023) on Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data)

**Research Papers:**

Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and
Rajiv Ramnath.
["A Countrywide Traffic Accident Dataset."](https://arxiv.org/abs/1906.05409), 2019.

Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu
Teodorescu, and Rajiv Ramnath.
["Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights."](https://arxiv.org/abs/1909.09638)
In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances
in Geographic Information Systems, ACM, 2019.
