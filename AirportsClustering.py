import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
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
file_path = resource_path("airports.csv")
data = pd.read_csv(file_path, na_values=["", " "])

# Data Preprocessing
# Drops rows with missing information for the selected features
cleaned_data = data[['ident', 'name', 'latitude_deg', 'longitude_deg', 'elevation_ft', 'iso_country','type']].dropna()
# Label encoder used to turn airport "type" into numerical label
cleaned_data['type_enc'] = LabelEncoder().fit_transform(cleaned_data['type'])
# Pick important features for the clustering 
features = cleaned_data[['latitude_deg', 'longitude_deg', 'elevation_ft']]
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
k = 5
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
    lat, lon, elev = original
    print(f"Cluster {i}: " f"{scaled}, " f"(Unscaled) Latitude = {lat:.4f}, Longitude = {lon:.4f}, Elevation = {elev:.2f}")

# Stops output data truncation from the describe()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Elevation Trends
# Mean is slightly different since describe() calculate from actual data points in cluster
# Original centroid values are the average of the standardized vectors, not original raw values, center shifts slightly when back projected into original space
elevation_trend = cleaned_data.groupby('cluster')['elevation_ft'].describe().to_string()
# Count is # of airports, mean is average elevation, std is how spread out elevations are, higher means more variation, min is lowest elevation(near sea or below if negative),
# 25% is lower quartile, 50% median means half airports lower and higher, 75% upper quartile, max is highest airport elevation in cluster
#print(elevation_trend)
# Save to a text file
with open("cluster_distribution.txt", "w") as file:
    file.write(elevation_trend)
# Interquartile range(IQR) = Q3-Q1
# Outliers calculated by Q3+1.5*IQR and Q1-1.5*IQR
# Lots of outliers still normal since elevation varies a lot globally, within 1 spatially broad cluster naturally gets wide variance and extreme points
# Ex: cluster with 90% lowland airports and 10% high elevation airports or ones with coastal and inland airports will have lots of statistical outliers
def elevation_boxplots(cleaned_data):
    # Group elevation data by cluster
    grouped_elevations = [
        group["elevation_ft"].dropna().values
        for _, group in cleaned_data.groupby("cluster")
    ]
    plt.figure(figsize=(12, 7))
    plt.boxplot(grouped_elevations, patch_artist=True, labels=[f"Cluster {i}" for i in range(len(grouped_elevations))])
    plt.title("Elevation Distribution of Clusters")
    plt.ylabel("Elevation (ft)")
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
        3: 'purple',
        4: 'brown'
    }
    # My 2D graph with latitude plotted against the longitude
    plt.figure(figsize=(12, 7))
    handles = []
    for cluster_id, color in cluster_colors.items():
        subset = cleaned_data[cleaned_data['cluster'] == cluster_id]
        plt.scatter(
            subset['longitude_deg'],
            subset['latitude_deg'],
            c=color,
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            edgecolor='k'
        )
        handles.append(mpatches.Patch(color=color, label=f'Cluster {cluster_id}'))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('K-Means Clustering of Airports')
    plt.grid(True)
    plt.legend(handles=handles)
    plt.tight_layout()
    plt.savefig("airport_2D.png")
    plt.close()

    # 3D graph with the introduction of elevation in ft as the z-axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    for cluster_id, color in cluster_colors.items():
        subset = cleaned_data[cleaned_data['cluster'] == cluster_id]
        ax.scatter(
            subset['longitude_deg'],
            subset['latitude_deg'],
            subset['elevation_ft'],
            c=color,
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            edgecolors='k'
        )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation (ft)')
    ax.set_title('K-Means Clustering of Airports with Elevation')
    legend_patches = [mpatches.Patch(color=color, label=f'Cluster {cluster_id}') for cluster_id, color in cluster_colors.items()]
    ax.legend(handles=legend_patches)
    plt.tight_layout()
    plt.savefig("airport_3D.png")
    plt.close()

cleaned_data.to_pickle("cleaned_data.pkl")

# Finding the most similar airport based on imput
# This function has been integrated into the user interface in MyMenu.py file
'''
def calculate_similarity(airport_id):
    # Notify user if input airport does not exist
    if airport_id not in cleaned_data['ident'].values:
        print(f"{airport_id} does not exist.")
        return None
    
    input_airport = cleaned_data[cleaned_data['ident'] == airport_id].iloc[0]
    cluster = cleaned_data['cluster']
    input_vector = np.array(input_airport['scaled'])
    # Grab all other points from the same cluster class as the input_airport
    cluster_class = cleaned_data[(cleaned_data['cluster'] == cluster) & (cleaned_data['ident'] != airport_id)].copy()
    cluster_vector = np.stack(cluster_class['scaled'].values)
    d = cdist([input_vector], cluster_vector)[0]
    #nearest = np.argmin(d)
    cluster_class['distance'] = d 

    # Calculations for distance of similar/farthest airport from the selected airport
    # Below has been moved to MyMenu.py
    
    nearest_airport = cluster_class.loc[cluster_class['distance'].idxmin()]
    farthest_airport = cluster_class.loc[cluster_class['distance'].idxmax()]
    print(f"\nSelected Airport: {input_airport['name']}, {input_airport['iso_country']} ({airport_id}) {input_airport['latitude_deg']} {input_airport['longitude_deg']} {input_airport['elevation_ft']}")
    print(f"Most similar Airport: {nearest_airport['name']}, {nearest_airport['iso_country']} ({nearest_airport['ident']}) {nearest_airport['latitude_deg']} {nearest_airport['longitude_deg']} {nearest_airport['elevation_ft']}")
    print(f"Distance: {nearest_airport['distance']:.4f}")
    print(f"Least similar Airport: {farthest_airport['name']}, {farthest_airport['iso_country']} ({farthest_airport['ident']}) {farthest_airport['latitude_deg']} {farthest_airport['longitude_deg']} {farthest_airport['elevation_ft']}")
    print(f"Distance: {farthest_airport['distance']:.4f}")
'''
elevation_boxplots(cleaned_data)
visualize_clusters()
#calculate_similarity("KLAS")