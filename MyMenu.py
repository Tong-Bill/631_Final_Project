import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd
import subprocess
import os
import sys
from sklearn.metrics.pairwise import euclidean_distances
# Referenced from tutorial: https://docs.python.org/3/library/tkinter.html
# Tkinter needs pip install Pillow to show images
# Maybe send input to temporary file, then ave airports clustering read the temp file for type of clustering and airport ID code inputted
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS 
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def run_script():
    # Run clustering script
    script_path = resource_path('AirportsClustering.py')
    result = subprocess.run(['python', script_path], capture_output=True, text=True, check=True)
    
    load_placeholder()

    # Show output
    output_text_1.config(state=tk.NORMAL)
    output_text_1.delete(1.0, tk.END)
    output_text_1.insert(tk.END, result.stdout)
    output_text_1.config(state=tk.DISABLED)
    
    output_text_2.config(state=tk.NORMAL)
    output_text_2.delete(1.0, tk.END)
    output_text_2.config(state=tk.DISABLED) 

    # Enable buttons after script has run
    btn_3d.config(state=tk.NORMAL)
    btn_4d.config(state=tk.NORMAL)
    btn_5d.config(state=tk.NORMAL)
    btn_6d.config(state=tk.NORMAL)
    btn_7d.config(state=tk.NORMAL)
    btn_8d.config(state=tk.NORMAL)
    btn_9d.config(state=tk.DISABLED)

def run_script_2():
    # Run clustering script
    result = subprocess.run(['python', 'RunwaysClustering.py'], capture_output=True, text=True, check=True)
    
    load_placeholder()

    # Show output
    output_text_1.config(state=tk.NORMAL)
    output_text_1.delete(1.0, tk.END)
    output_text_1.insert(tk.END, result.stdout)
    output_text_1.config(state=tk.DISABLED) 

    output_text_2.config(state=tk.NORMAL)
    output_text_2.delete(1.0, tk.END)
    output_text_2.config(state=tk.DISABLED) 

    # Enable buttons after script has run
    btn_3d.config(state=tk.NORMAL)
    btn_4d.config(state=tk.NORMAL)
    btn_5d.config(state=tk.DISABLED)
    btn_6d.config(state=tk.DISABLED)
    btn_7d.config(state=tk.NORMAL)
    btn_8d.config(state=tk.NORMAL)
    btn_9d.config(state=tk.NORMAL)

# Load placeholder at startup
def load_placeholder():
    img_path = resource_path("DEM_topo.png")
    img = Image.open(img_path)
    img = img.resize((800, 500), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

def show_image(path):
    img = Image.open(path)
    img = img.resize((800, 500), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

def calculate_similarity(airport_id, df):
    try:
        target = df[df['ident'] == airport_id]
        if target.empty:
            return "Airport ID not found."

        target = target.iloc[0]
        # Ensure features are clean
        features = df[['latitude_deg', 'longitude_deg', 'elevation_ft']].dropna()
        target_feat = target[['latitude_deg', 'longitude_deg', 'elevation_ft']].values.reshape(1, -1)

        df = df.copy()
        df = df.loc[features.index]  # align df with cleaned features
        df['distance'] = euclidean_distances(df[['latitude_deg', 'longitude_deg', 'elevation_ft']], target_feat)

        # Cluster ID of input
        cluster_id = target['cluster']

        # Elevation comparison within cluster
        cluster_subset = df[df['cluster'] == cluster_id]
        max_elev = cluster_subset['elevation_ft'].max()
        min_elev = cluster_subset['elevation_ft'].min()
        above_lowest = target['elevation_ft'] - min_elev
        below_highest = max_elev - target['elevation_ft']

        # Filter: same cluster (excluding self)
        same_cluster_df = df[(df['cluster'] == cluster_id) & (df['ident'] != airport_id)]
        most_similar = same_cluster_df.loc[same_cluster_df['distance'].idxmin()] if not same_cluster_df.empty else None
        least_similar = same_cluster_df.loc[same_cluster_df['distance'].idxmax()] if not same_cluster_df.empty else None

        # Filter: all other airports
        all_others_df = df[df['ident'] != airport_id]
        least_similar_g = all_others_df.loc[all_others_df['distance'].idxmax()]

        # Get most similar and least similar
        # Distance is how far away from the selected airport, it is a hybrid unit combining degrees of latitude, longitude, and feet of elevation
        output = f'''
        \nSelected Airport: {target['name']}, {target['iso_country']} ({airport_id}) \nLatitude:{target['latitude_deg']} Longitude:{target['longitude_deg']} Elevation:{target['elevation_ft']} \nCluster: {target['cluster']}
        \nComparison Within Cluster {cluster_id}:\nElevation of highest airport in cluster: {max_elev:.1f} ft \nElevation of lowest airport in cluster: {min_elev:.1f} ft \nSelected airport is {below_highest:.1f} ft below the highest \nSelected airport is {above_lowest:.1f} ft above the lowest
        \nMost similar: {most_similar['name']}, {most_similar['iso_country']} ({most_similar['ident']}) \nLatitude:{most_similar['latitude_deg']} Longitude:{most_similar['longitude_deg']} Elevation:{most_similar['elevation_ft']} \nCluster: {most_similar['cluster']} Distance: {most_similar['distance']:.4f}
        \nLeast similar: {least_similar['name']}, {least_similar['iso_country']} ({least_similar['ident']}) \nLatitude:{least_similar['latitude_deg']} Longitude:{least_similar['longitude_deg']} Elevation:{least_similar['elevation_ft']} \nCluster: {least_similar['cluster']} Distance: {least_similar['distance']:.4f}
        \nLeast similar Globally: {least_similar_g['name']}, {least_similar_g['iso_country']} ({least_similar_g['ident']}) \nLatitude:{least_similar_g['latitude_deg']} Longitude:{least_similar_g['longitude_deg']} Elevation:{least_similar_g['elevation_ft']} \nCluster: {least_similar_g['cluster']} Distance: {least_similar_g['distance']:.4f}
        '''
        return output
    except Exception as e:
        return f"Error: {str(e)}"

def simulation_window():
    sim_window = tk.Toplevel(root)
    sim_window.title("Airport Simulation")
    sim_window.geometry("600x580")

    tk.Label(sim_window, text="Enter Airport Identifier:", font=("Arial", 12)).pack(pady=(20, 5))
    airport_entry = tk.Entry(sim_window, font=("Courier New", 10))
    airport_entry.pack(pady=5)
    result_label = tk.Label(sim_window, text="", font=("Courier New", 10), wraplength=550, justify="left")
    result_label.pack(pady=20)

    def run_similarity():
        # Load cleaned_data.pkl when the GUI starts
        try:
            file_path = resource_path("cleaned_data.pkl")
            cleaned_data = pd.read_pickle(file_path)
        except FileNotFoundError:
            cleaned_data = None
        airport_id = airport_entry.get().strip()
        
        if not airport_id:
            result = "Invalid airport identifier code."
        elif cleaned_data is None:
            result = "cleaned_data.pkl not loaded"
        else:
            result = calculate_similarity(airport_id, cleaned_data)
        result_label.config(text=result)

    submit_btn = tk.Button(sim_window, text="Run Simulation", command=run_similarity)
    submit_btn.pack(pady=5)
    close_btn = tk.Button(sim_window, text="Quit", command=sim_window.destroy)
    close_btn.pack(pady=10)

def calculate_surface(length, width, cluster_id, df):
    # Calculates euclidean distance between input and each runway in cluster
    # Takes the 10 closest runway, sees what surface type they have and frequency, then gives probabilities 
    cluster_df = df[df['cluster'] == cluster_id]
    if cluster_df.empty:
        return "Error with data"
    cluster_df = cluster_df.copy()
    cluster_df['distance'] = ((cluster_df['length_ft'] - length)**2 + (cluster_df['width_ft'] - width)**2) ** 0.5
    # Select N closest runways (e.g., 10)
    top_n = cluster_df.nsmallest(10, 'distance')
    # Compute surface type probabilities
    surface_probs = top_n['surface'].value_counts(normalize=True).round(3) * 100
    result = "Surface Type Probabilities:\n"
    for surface, prob in surface_probs.items():
        result += f"• {surface}: {prob:.1f}%\n"
    return result

def simulation_window_2():
    #Ideas:
    # Maybe input length and width, outputs probability of each surface type runway in each cluster
    sim_window = tk.Toplevel(root)
    sim_window.title("Runway Simulation")
    sim_window.geometry("600x580")

    # Labels and entries
    tk.Label(sim_window, text="Enter Runway Information:", font=("Arial", 12)).pack(pady=(20, 5))

    tk.Label(sim_window, text="Length (ft):").pack(pady=5)
    entry_length = tk.Entry(sim_window)
    entry_length.pack(pady=5)

    tk.Label(sim_window, text="Width (ft):").pack(pady=5)
    entry_width = tk.Entry(sim_window)
    entry_width.pack(pady=5)

    tk.Label(sim_window, text="Cluster Number:").pack(pady=5)
    entry_cluster = tk.Entry(sim_window)
    entry_cluster.pack(pady=5)

    result_label = tk.Label(sim_window, text="", font=("Courier", 10), justify="left", wraplength=450)
    result_label.pack(pady=15)

    def run_simulation():
        try:
            file_path = resource_path("cleaned_data.pkl")
            cleaned_data = pd.read_pickle(file_path)
            length = float(entry_length.get())
            width = float(entry_width.get())
            cluster = int(entry_cluster.get())

            if cleaned_data is None:
                result = "Runway data not loaded."
            else:
                result = calculate_surface(length, width, cluster, cleaned_data)
        except ValueError:
            result = "Invalid Values"

        result_label.config(text=result)

    tk.Button(sim_window, text="Run Simulation", command=run_simulation).pack(pady=10)
    tk.Button(sim_window, text="Quit", command=sim_window.destroy).pack(pady=5)

# --- UI Setup ---
root = tk.Tk()
root.title("Airport Clustering Console")
root.geometry("1200x800")

# --- Top Layout Container (Sidebar + Main Content) ---
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# --- Sidebar + Main Content Setup ---
sidebar = tk.Frame(top_frame, width=200, bg='#eeeeee')
sidebar.pack(side=tk.LEFT, fill=tk.Y)
main_area = tk.Frame(top_frame)
main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# --- Sidebar Content ---
tk.Label(sidebar, text="Actions", font=("Arial", 14, "bold"), bg='#eeeeee').pack(pady=5)

run_btn = tk.Button(sidebar, text="Geographic Clustering", fg="red", command=run_script)
run_btn.pack(pady=5, fill=tk.X, padx=5)

btn_2d = tk.Button(sidebar, text="Runway Clustering", fg="red", command=run_script_2)
btn_2d.pack(pady=5, fill=tk.X, padx=5)

img2_path = resource_path("elbow_image.png")
btn_3d = tk.Button(sidebar, text="Elbow Method", state=tk.DISABLED, command=lambda: show_image(img2_path))
btn_3d.pack(pady=5, fill=tk.X, padx=5)

img3_path = resource_path("clusters_boxplot.png")
btn_4d = tk.Button(sidebar, text="Cluster Distribution", state=tk.DISABLED, command=lambda: (
        show_image(img3_path),
        output_text_2.config(state=tk.NORMAL),
        output_text_2.delete(1.0, tk.END),
        output_text_2.insert(tk.END, open("cluster_distribution.txt").read()),
        output_text_2.config(state=tk.DISABLED)
    ))
btn_4d.pack(pady=5, fill=tk.X, padx=5)

btn_5d = tk.Button(sidebar, text="Airport Simulation", state=tk.DISABLED, command=simulation_window)
btn_5d.pack(pady=5, fill=tk.X, padx=5)

img_path = resource_path("DEM_topo.png")
btn_6d = tk.Button(sidebar, text="Global Elevation", state=tk.DISABLED, command=lambda: show_image(img_path))
btn_6d.pack(pady=5, fill=tk.X, padx=5)

img4_path = resource_path("airport_2D.png")
btn_7d = tk.Button(sidebar, text="Show 2D Plot", state=tk.DISABLED, command=lambda: show_image(img4_path))
btn_7d.pack(pady=5, fill=tk.X, padx=5)

img5_path = resource_path("airport_3D.png")
btn_8d = tk.Button(sidebar, text="Show 3D Plot", state=tk.DISABLED, command=lambda: show_image(img5_path))
btn_8d.pack(pady=5, fill=tk.X, padx=5)

btn_9d = tk.Button(sidebar, text="Runway Simulation", state=tk.DISABLED, command=simulation_window_2)
btn_9d.pack(pady=5, fill=tk.X, padx=5)

# --- Main Area Split ---
content_frame = tk.Frame(main_area)
content_frame.pack(fill=tk.BOTH, expand=True)

# Image label (left of content_frame)
image_label = tk.Label(content_frame)
image_label.pack(side=tk.LEFT, padx=10, pady=10, expand=True)

# --- OUTPUT AREA (Vertical Split) ---
# --- BOTTOM OUTPUT AREA (aligned with main area) ---
bottom_frame = tk.Frame(root)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=5)

# Spacer to match sidebar width (so output doesn't go under it)
sidebar_spacer = tk.Frame(bottom_frame, width=150)
sidebar_spacer.pack(side=tk.LEFT)

# Main output wrapper (right of sidebar)
output_wrapper = tk.Frame(bottom_frame)
output_wrapper.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# --- Cluster Means Section ---
cluster_frame = tk.Frame(output_wrapper)
cluster_frame.pack(fill=tk.BOTH, expand=True)

cluster_scroll = tk.Scrollbar(cluster_frame)
cluster_scroll.pack(side=tk.RIGHT, fill=tk.Y)

output_text_1 = tk.Text(
    cluster_frame,
    wrap=tk.WORD,
    font=("Courier New", 10),
    yscrollcommand=cluster_scroll.set,
    height=6,
    bg=output_wrapper.cget("bg"),   
    bd=0,                           
    highlightthickness=0,           
    relief="flat",                  
    state=tk.DISABLED               
)
output_text_1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
cluster_scroll.config(command=output_text_1.yview)

console_frame = tk.Frame(output_wrapper)
console_frame.pack(fill=tk.BOTH, expand=True)

console_scroll = tk.Scrollbar(console_frame)
console_scroll.pack(side=tk.RIGHT, fill=tk.Y)

output_text_2 = tk.Text(
    console_frame,
    wrap=tk.WORD,
    font=("Courier New", 10),
    yscrollcommand=console_scroll.set,
    height=8,
    bg=output_wrapper.cget("bg"),   
    bd=0,                           
    highlightthickness=0,           
    relief="flat",                  
    state=tk.DISABLED               
)
output_text_2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
console_scroll.config(command=output_text_2.yview)

load_placeholder()

# --- Start GUI ---
root.mainloop()