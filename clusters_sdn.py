import numpy as np
import matplotlib.pyplot as plt
from numpy import *
from operator import truediv 
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter

col_list = ["pktperflow", "byteperflow", "label", "Protocol"]

df = pd.read_csv("dataset_sdn.csv", usecols=col_list, sep=',')

x = df["pktperflow"]
y = df["byteperflow"]
z = df["label"]
q = df["Protocol"]

xx = list(x)
yy = list(y)
zz = list(z)
qq = list(q)

xx = xx[:100000]
yy = yy[:100000]
zz = zz[:100000]
qq = qq[:100000]

for j in qq[:]:
   if j == "ICMP" or j == "UDP":
      del xx[qq.index(j)]
      del yy[qq.index(j)]
      del zz[qq.index(j)]
      qq.remove(j)

for j in xx[:]:
   if j<0:
      del yy[xx.index(j)]
      del zz[xx.index(j)]
      del qq[xx.index(j)]
      xx.remove(j)

for j in yy[:]:
   if j<0:
      del xx[yy.index(j)]
      del zz[yy.index(j)]
      del qq[yy.index(j)]
      yy.remove(j)

for j in zz[:]:
   if j==0:
      del xx[zz.index(j)]
      del yy[zz.index(j)]
      del qq[zz.index(j)]
      zz.remove(j)

df = list(zip(xx, yy))
for i in df:
   if i[0]<0 or i[1]<0:
      print(i)

pca = PCA(2)
df = pca.fit_transform(df)

kmeans = KMeans(n_clusters= 2)
label = kmeans.fit_predict(df)
a = dict(Counter(label))
print(a)
#filter rows of original data
u_labels = np.unique(label)
centroids = kmeans.cluster_centers_
#plt.ylim([0, 500])
#plt.xlim([0, 2000])
#plotting the results:
plt.title(a)
 
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = "black")
plt.legend()
plt.show()

t = set(yy)
c = len(t)

for i in set(yy):
   plt.axhline(y=i)
plt.title(c)
plt.plot(xx, yy, 'o', color='black')
#plt.ylim([0, 500])
#plt.xlim([0, 2000])
plt.show()