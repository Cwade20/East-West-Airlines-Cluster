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

