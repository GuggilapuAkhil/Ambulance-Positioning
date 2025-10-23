import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import folium

# ---------------------- Step 1: Load and Preprocess Data ----------------------
print("Loading dataset...")
df = pd.read_csv('accidents.csv')

# Drop rows with missing coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])

# Convert 'Time' to hour of day
df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour

# Encode 'Weather_Condition' as categorical codes
df['Weather_Condition'] = df['Weather_Condition'].astype('category').cat.codes

# Drop rows with NaN in converted columns
df = df.dropna(subset=['Hour'])

# ---------------------- Step 2: Define Features and Target ----------------------
features = df[['Latitude', 'Longitude', 'Hour', 'Weather_Condition']]
target = df['Severity']

# ---------------------- Step 3: Train-Test Split and Model ----------------------
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

print("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Print Evaluation Metrics
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ---------------------- Step 4: KMeans for Ambulance Location ----------------------
print("Running KMeans clustering to determine ambulance positions...")
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

# ---------------------- Step 5: Generate Interactive Map ----------------------
print("Generating map with accident points and ambulance locations...")
center_location = [df['Latitude'].mean(), df['Longitude'].mean()]
m = folium.Map(location=center_location, zoom_start=12)

# Plot accident points
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.6
    ).add_to(m)

# Plot suggested ambulance center locations
for i, center in enumerate(kmeans.cluster_centers_):
    folium.Marker(
        location=center,
        popup=f"Ambulance Location #{i+1}",
        icon=folium.Icon(color='blue', icon='plus-sign')
    ).add_to(m)

# Save and notify
map_path = "ambulance_positions.html"
m.save(map_path)
print(f"\n✅ Map successfully saved as '{map_path}'")
