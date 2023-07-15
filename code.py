from ast import increment_lineno
import pandas as pd
from sklearn.cluster import KMeans
import folium
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline



df = pd.read_csv("/content/drive/MyDrive/report for clustering.csv")
df.head()


x = df[["Lattitude","Longitude"]]
x.head()


model = KMeans(n_clusters=3)
y_kmeans = model.fit_predict(x)



df['y'] = y_kmeans


df.head()

plt.scatter(df['Longitude'], df['Lattitude'],c=df['y'],)
plt.scatter(df['Longitude'], df['Lattitude'],c=df['Accident Severity Index'])
wcss = []
for i in range(1,11):
    model = KMeans(n_clusters=i)
    y_kmeans = model.fit_predict(x)
    wcss.append(model.inertia_)
model.inertia_
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
cluster1 = df[['Lattitude', "Longitude"]][df['y'] == 0].values.tolist()
cluster2 = df[['Lattitude', "Longitude"]][df['y'] == 1].values.tolist()
cluster3 = df[['Lattitude', "Longitude"]][df['y'] == 2].values.tolist()
kerala_map = folium.Map(location=[10.8505, 76.2711], zoom_start=8,tiles = "openstreetmap")
kerala_map
for i in cluster1:
    folium.CircleMarker(i, radius=2,color='blue',fill_color='lightblue').add_to(kerala_map)

for i in cluster2:
    folium.CircleMarker(i, radius=2,color='red',fill_color='lightred').add_to(kerala_map)

for i in cluster3:
    folium.CircleMarker(i, radius=2,color='green',fill_color='lightgreen').add_to(kerala_map)
kerala_map

