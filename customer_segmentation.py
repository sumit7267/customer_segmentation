import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

customer = pd.read_csv("Mall_Customers.csv")

#print(customer.head())

#preprocessing 
print(customer.shape)   # for number of rows and features
print(customer.isnull().sum())   #for checking null fields 
print(customer.duplicated().sum())  #for checking duplicates
print(customer.info())   #for overall info
print(customer.describe())  #for statistics of data

#EDA
# sns.set()
# sns.pairplot(customer)
# plt.show()

# plt.figure(figsize=(6,6))
# sns.histplot(customer['Age'])
# plt.show()

x=customer.iloc[:,[3,4]].values  # because we want only annual income and spending score

#applying KMeans algorithm

#finding optimal number of clusters
wcss = []
for i in range(1,11):
  k_means = KMeans(n_clusters=i , init='k-means++', random_state=45)
  k_means.fit(x)
  wcss.append(k_means.inertia_)

plt.plot(range(1,11), wcss)
plt.show() # from plot we got optimal number of clusters as 7


#applying Kmeans algorithm for 7 clusters
kmeans_ = KMeans(n_clusters=7, init='k-means++', random_state=0)

y = kmeans_.fit_predict(x)

plt.figure(figsize=(8,8))
plt.scatter(x[y==0,0] , x[y==0,1], s=50, c='green', label='cluster1')
plt.scatter(x[y==1,0] , x[y==1,1], s=50, c='yellow', label='cluster2')
plt.scatter(x[y==2,0] , x[y==2,1], s=50, c='red', label='cluster3')
plt.scatter(x[y==3,0] , x[y==3,1], s=50, c='blue', label='cluster4')
plt.scatter(x[y==4,0] , x[y==4,1], s=50, c='black', label='cluster5')
plt.scatter(x[y==5,0] , x[y==5,1], s=50, c='orange', label='cluster6')
plt.scatter(x[y==6,0] , x[y==6,1], s=50, c='purple', label='cluster7')

#plotting centroids
plt.scatter(kmeans_.cluster_centers_[:,0], kmeans_.cluster_centers_[:,1], s=100, c='cyan', label='centroid')

plt.title("Customer Clustering")
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


