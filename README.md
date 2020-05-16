# East-West-Airlines-Cluster
## In this exercise, your goal is to use cluster analysis to identify segments of passengers that have similar characteristics for the purpose of targeting different segments for different types of mileage offers. 
Apply k-Means clustering using all the attributes (except ID#) with k = 2 through k = 6. For each k, record the Davies Bouldin Index.

(Hint: Remember to normalize the variables before clustering.)

List the DBI for different values of k. Based on the DBI metric, how many clusters of passengers are there?

Describe the characteristics of the passengers in each cluster for your chosen number of clusters (i.e., for the best k)

Based on the characteristics of the clusters, give them appropriate names.


```python
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import train_test_split

df= pd.read_excel("C:/Users/Charles/Downloads/EastWestAirlines.xls",sheet_name="data")

df.isnull().sum()

>ID#                  0
Balance              0
Qual_miles           0
cc1_miles            0
cc2_miles            0
cc3_miles            0
Bonus_miles          0
Bonus_trans          0
Flight_miles_12mo    0
Flight_trans_12      0
Days_since_enroll    0
Award?               0
dtype: int64

X= df.iloc[:,1:]
X.head()
```
![image](https://user-images.githubusercontent.com/61456930/82120111-ca46bc80-9751-11ea-9f51-466419f133b4.PNG)

```python

scaler=MinMaxScaler()

features = [['Balance','Qual_miles','Bonus_miles','Flight_miles_12mo','Days_since_enroll']]
features
```
```
[['Balance',
  'Qual_miles',
  'Bonus_miles',
  'Flight_miles_12mo',
  'Days_since_enroll']]
  
  for feature in features:
    df[feature] = scaler.fit_transform(df[feature])
    
```
![image](https://user-images.githubusercontent.com/61456930/82120281-e565fc00-9752-11ea-8e28-b50bf49cd402.PNG)

```python

for i in range(2,7):
    kmeans = KMeans(n_clusters=i, random_state=1).fit(X)
    labels = kmeans.labels_
    db=davies_bouldin_score(X, labels)
    print("Davies Bouldin Index for",i," neighbours is",db)
    
```

```
Davies Bouldin Index for 2  neighbours is 0.6322390890026537
Davies Bouldin Index for 3  neighbours is 0.6281037424524417
Davies Bouldin Index for 4  neighbours is 0.6209677377393183
Davies Bouldin Index for 5  neighbours is 0.6601845416287746
Davies Bouldin Index for 6  neighbours is 0.6451947320940435
```

```python

from sklearn import preprocessing
dataset1_standardized = preprocessing.scale(df)
dataset1_standardized = pd.DataFrame(dataset1_standardized)
````

```python
plt.figure(figsize=(10, 8))
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 6):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset1_standardized)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 6), wcss)
plt.title('KMeans')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```
![image](https://user-images.githubusercontent.com/61456930/82131539-2a664e80-97a4-11ea-9432-a12487b19a7e.PNG)

```python
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(dataset1_standardized)
#beginning of  the cluster numbering with 1 instead of 0
y_kmeans1=y_kmeans
y_kmeans1=y_kmeans+1
# New Dataframe called cluster
cluster = pd.DataFrame(y_kmeans1)
# Adding cluster to the Dataset1
df['cluster'] = cluster
#Mean of clusters
kmeans_mean_cluster = pd.DataFrame(round(df.groupby('cluster').mean(),1))
kmeans_mean_cluster
```
![image](https://user-images.githubusercontent.com/61456930/82131561-6f8a8080-97a4-11ea-8240-4effcf1d361e.PNG)

```python
# Hierarchical clustering for the same dataset
# creating a dataset for hierarchical clustering
dataset2_standardized = dataset1_standardized
# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
# some setting for this notebook to actually show the graphs inline
%matplotlib inline
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
#creating the linkage matrix
H_cluster = linkage(dataset2_standardized,'ward')
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    H_cluster,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=5,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()
```
![image](https://user-images.githubusercontent.com/61456930/82131675-a8772500-97a5-11ea-98ec-656198f50c54.PNG)

```python
# Assigning the clusters and plotting the observations as per hierarchical clustering
from scipy.cluster.hierarchy import fcluster
k=5
cluster_2 = fcluster(H_cluster, k, criterion='maxclust')
cluster_2[0:30:,]
plt.figure(figsize=(10, 8))
plt.scatter(dataset2_standardized.iloc[:,0], dataset2_standardized.iloc[:,1],c=cluster_2, cmap='prism')  # plot points with cluster dependent colors
plt.title('Airline Data - Hierarchical Clutering')
plt.show()
```
![image](https://user-images.githubusercontent.com/61456930/82131676-ae6d0600-97a5-11ea-8abb-b3b6773014fa.PNG)

```python
# New Dataframe called cluster
cluster_Hierarchical = pd.DataFrame(cluster_2)
# Adding the hierarchical clustering to dataset
dataset2=df
dataset2['cluster'] = cluster_Hierarchical
dataset2.head()
```
![image](https://user-images.githubusercontent.com/61456930/82131677-b0cf6000-97a5-11ea-9924-4efb2a9d14f1.PNG)
