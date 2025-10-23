import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import folium

# Load dataset
df = pd.read_csv('accidents.csv')
df = df.dropna(subset=['Latitude', 'Longitude'])

# Encode categorical features
df['Weather_Condition'] = df['Weather_Condition'].astype('category').cat.codes
df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour

# Feature matrix and target
features = df[['Latitude', 'Longitude', 'Hour', 'Weather_Condition']]
target = df['Severity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Print evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# KMeans for ambulance positioning
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])

# Create map
center_location = [df['Latitude'].mean(), df['Longitude'].mean()]
m = folium.Map(location=center_location, zoom_start=13)

# Plot accidents
for _, row in df.iterrows():
    folium.CircleMarker(location=[row['Latitude'], row['Longitude']],
                        radius=3, color='red', fill=True).add_to(m)

# Plot suggested ambulance locations
for center in kmeans.cluster_centers_:
    folium.Marker(location=center, icon=folium.Icon(color='blue', icon='plus')).add_to(m)

# Save map
m.save("ambulance_positions.html")
print("Map saved as 'ambulance_positions.html'")
