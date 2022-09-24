import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 20)

cust_df = pd.read_csv("CSV/Clustering/Cust_Segmentation.csv")
cust_df = cust_df.drop('Address', axis=1)
print(cust_df)

X = cust_df.values[:, 1:]
X = np.nan_to_num(X)
dataset = StandardScaler().fit_transform(X)

Cluster_num = 3
k_means = KMeans(init="k-means++", n_clusters=Cluster_num, n_init=12)
k_means.fit(dataset)
labels = k_means.labels_

cust_df['class'] =labels

print(cust_df.groupby('class').mean())

