import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
# Data & Resources: https://ourairports.com/data/#excel, https://ourairports.com/help/data-dictionary.html
# Retrieved picture: https://ggos.org/item/dem-digital-elevation-model/

# Absolute path for executable, not needed anymore
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load the data from the airport csv
# Empty strings in data need to be viewed as NaN
file_path = resource_path("runways.csv")
data = pd.read_csv(file_path, na_values=["", " "])

# Data Preprocessing
# Drops rows with missing information for the selected features
cleaned_data = data[['airport_ident', 'length_ft', 'width_ft', 'surface']].dropna()

# unique_surfaces = data['surface'].dropna().unique()
# Keywords for most common surface types
surface_map = {
    'hard': ['ASPH', 'CONC', 'CONCRETE', 'BIT', 'PAVED', 'ASPHALT', 'TARMAC'],
    'medium': ['GRVL', 'GRAVEL', 'OIL', 'BRICK', 'LATERITE', 'MACADAM', 'HARDCORE'],
    'soft': ['GRASS', 'TURF', 'DIRT', 'CLAY', 'SOD', 'SOIL', 'EARTH', 'LOAM'],
    'very_soft': ['SAND', 'MUD', 'SLUSH', 'SHALE', 'LIMESTONE'],
    'water': ['WATER', 'ICE', 'SNOW', 'LAKE'],
}

# Convert surface type to a numerical scale from 0 to 1 in hardness
# 0 represents very fluid i.e. water, 1 represents very hard i.e. concrete
# Unmapped surfaces by default are set to 0.0
def surface_score(surface):
    if pd.isna(surface):
        return 0.0

    surf = str(surface).upper()

    for keyword in surface_map['hard']:
        if keyword in surf:
            return 1.0
    for keyword in surface_map['medium']:
        if keyword in surf:
            return 0.6
    for keyword in surface_map['soft']:
        if keyword in surf:
            return 0.4
    for keyword in surface_map['very_soft']:
        if keyword in surf:
            return 0.2
    for keyword in surface_map['water']:
        if keyword in surf:
            return 0.1
    return 0.0

cleaned_data['surface_enc'] = cleaned_data['surface'].apply(surface_score).fillna(0.0)
# Pick important features for the clustering 
features = cleaned_data[['length_ft', 'width_ft', 'surface_enc']].dropna()
# Normalize the features using the StandardScaler class from sklearn
features_scaled = StandardScaler().fit_transform(features)

# Elbow Method used below to determine optimal clusters(k) to use
# Inertia or Within-Cluster Sum of Squares(WCSS) quantifies how tightly clustered data points are in each cluster
# Taken from https://www.w3schools.com/python/python_ml_k-means.asp 
inertia = []
for i in range(2, 10):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(features_scaled)
    inertia.append(km.inertia_)

plt.plot(range(2, 10), inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.savefig("elbow_image.png")

# Clustering process
# K value taken from elbow method above
k = 4
model = KMeans(n_clusters=k, random_state=42)
cleaned_data['cluster'] = model.fit_predict(features_scaled)
cleaned_data['scaled'] = list(features_scaled)

# Calculating Scaled Centroids(mean)
# Calculations for Unscaled Centroids(mean)
scaler = StandardScaler()
scaler.fit(features)
# First get both versions of centroids
scaled_centroids = model.cluster_centers_
original_centroids = scaler.inverse_transform(scaled_centroids)
for i, (scaled, original) in enumerate(zip(scaled_centroids, original_centroids)):
    length, width, surf = original
    print(f"Cluster {i}: " f"{scaled}, " f"(Unscaled) Length = {length:.4f}, Width = {width:.4f}, Surface = {surf:.2f}")
# Stops output data truncation from the describe()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Runway length Trends
# Mean is slightly different since describe() calculate from actual data points in cluster
# Original centroid values are the average of the standardized vectors, not original raw values, center shifts slightly when back projected into original space
cleaned_data['runway_length'] = cleaned_data['length_ft']
area_trend = cleaned_data.groupby('cluster')['runway_length'].describe().to_string()
# In this context, mean is average length of runway
# Max is longest runway in the cluster, 25%, 50%, 75% statistics are also shown
#print(area_trend)
# Save to a text file
with open("cluster_distribution.txt", "w") as file:
    file.write(area_trend)
# Interquartile range(IQR) = Q3-Q1
# Outliers calculated by Q3+1.5*IQR and Q1-1.5*IQR
# Notice that there isn't as much outliers since we aren't dealing with elevation data
# Variation in length is not as skewed as width 

def area_boxplots(cleaned_data):
    # Group area data by cluster
    grouped_lengths = [
        group["runway_length"].dropna().values
        for _, group in cleaned_data.groupby("cluster")
    ]
    plt.figure(figsize=(12, 7))

    plt.boxplot(grouped_lengths, patch_artist=True, labels=[f"Cluster {i}" for i in range(len(grouped_lengths))])
    plt.title("Size Distribution of Clusters")
    plt.ylabel("Length (ft)")
    plt.xlabel("Cluster Numbers")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("clusters_boxplot.png") 

def visualize_clusters():
    # Custom cluster colors
    cluster_colors = {
        0: 'blue',
        1: 'green',
        2: 'orange',
        3: 'purple'
    }
    # My 2D graph with length plotted against the width of runways
    plt.figure(figsize=(12, 7))
    handles = []
    for cluster_id, color in cluster_colors.items():
        subset = cleaned_data[cleaned_data['cluster'] == cluster_id]
        plt.scatter(
            subset['width_ft'],
            subset['length_ft'],
            c=color,
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            edgecolor='k'
        )
        handles.append(mpatches.Patch(color=color, label=f'Cluster {cluster_id}'))
    plt.xlabel('width (ft)')
    plt.ylabel('Length (ft)')
    plt.title('K-Means Clustering of Runway Size')
    plt.grid(True)
    plt.legend(handles=handles)
    plt.tight_layout()
    plt.savefig("airport_2D.png")
    plt.close()

    # 3D graph with introduction of surface type scaled from 0 to 1 measureing hardness
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cluster_id, color in cluster_colors.items():
        subset = cleaned_data[cleaned_data['cluster'] == cluster_id]
        ax.scatter(
            subset['width_ft'],
            subset['length_ft'],
            subset['surface_enc'],
            c=color,
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            edgecolors='k'
        )
    ax.set_xlabel('width (ft)')
    ax.set_ylabel('length (ft)')
    ax.set_zlabel('Surface Type (numerical)')
    ax.set_title('K-Means Clustering of Airports with Surface')
    legend_patches = [mpatches.Patch(color=color, label=f'Cluster {cluster_id}') for cluster_id, color in cluster_colors.items()]
    ax.legend(handles=legend_patches)
    plt.tight_layout()
    plt.savefig("airport_3D.png")
    plt.close()

cleaned_data.to_pickle("cleaned_data.pkl")
area_boxplots(cleaned_data)
visualize_clusters()