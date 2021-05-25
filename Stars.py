########### TO-DO LIST ###########

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

strs = pd.read_csv("./Stars.csv")
strs.rename(index=str, columns={'L': 'Luminosity',
                                'R': 'Radius',
                                'A_M': 'Absolute Magnitude'}, inplace=True)

print(strs.head(20))

X = strs.drop(['Color', 'Spectral_Class'], axis=1)
sns.pairplot(data=X,hue="Type", palette=sns.color_palette('bright', 6))
plt.show()
sns.barplot(x="Type",y="Temperature",data=strs, palette=sns.color_palette('dark', 6))
plt.show()

###################### KMeans Graph - Elbow ######################

from sklearn.cluster import KMeans

clusters = []

for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)
    
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Searching for Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# Annotate arrow
ax.annotate('Possible Elbow Point', xy=(2, 2500000000000), xytext=(3, 5000000000000), xycoords='data',          
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

ax.annotate('Possible Elbow Point', xy=(3, 700000000000), xytext=(5, 1000000000000), xycoords='data',          
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

plt.show()

###################### KMeans 3-5-10 ######################

from sklearn.cluster import KMeans

# 3 cluster
km3 = KMeans(n_clusters=3).fit(X)

X['Labels'] = km3.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', 3))
plt.title('KMeans with 3 Clusters')
plt.show()

# Let's see with 5 Clusters
km6 = KMeans(n_clusters=6).fit(X)

X['Labels'] = km6.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', 6))
plt.title('KMeans with 6 Clusters')
plt.show()

# Let's see with 10 Clusters
km10 = KMeans(n_clusters=10).fit(X)

X['Labels'] = km10.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', 10))
plt.title('KMeans with 10 Clusters')
plt.show()

###################### OPTICS ######################

from sklearn.cluster import OPTICS 

optics = OPTICS(eps=0.5, min_samples=20).fit(X)

X['Labels'] = optics.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', np.unique(optics.labels_).shape[0]))
plt.title('OPTICS with ' + str(np.unique(optics.labels_).shape[0]) + ' Clusters')
plt.show()


###################### Affinity Propagation ######################

from sklearn.cluster import AffinityPropagation

affi = AffinityPropagation(damping=0.9).fit(X)

X['Labels'] = affi.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', np.unique(affi.labels_).shape[0]))
plt.title('Affinity Prop with '+ str(np.unique(affi.labels_).shape[0]) + ' Clusters')
plt.show()

###################### Agglomerative ######################

from sklearn.cluster import AgglomerativeClustering 

agglom = AgglomerativeClustering(n_clusters=5, linkage='average').fit(X)

X['Labels'] = agglom.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', 5))
plt.title('Agglomerative with ' + str(np.unique(agglom.labels_).shape[0]) + ' Clusters')
plt.show()

###################### Hierarchy dendogram ######################

from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 

dist = distance_matrix(X, X)

Z = hierarchy.linkage(dist, 'complete')

plt.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')

Z = hierarchy.linkage(dist, 'average')
plt.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

###################### DBSCAN ######################

from sklearn.cluster import DBSCAN 

db = DBSCAN(eps=50, min_samples=6).fit(X)

X['Labels'] = db.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', np.unique(db.labels_).shape[0]))
plt.title('DBSCAN with epsilon 50, min samples 6')
plt.show()

###################### MeanShift ######################

from sklearn.cluster import MeanShift, estimate_bandwidth

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.5)
ms = MeanShift(bandwidth).fit(X)

X['Labels'] = ms.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', np.unique(ms.labels_).shape[0]))
plt.plot()
plt.title('MeanShift with' + str(np.unique(ms.labels_).shape[0]) + ' Clusters' )
plt.show()

###################### Mini Batch K Means ######################

from sklearn.cluster import MiniBatchKMeans

mbk = MiniBatchKMeans(n_clusters=5).fit(X)

X['Labels'] = mbk.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', np.unique(mbk.labels_).shape[0]))
plt.title('Mini-batch with ' + str(np.unique(mbk.labels_).shape[0]) + ' clusters')
plt.show()

###################### Birch ######################

from sklearn.cluster import Birch

brch = Birch(threshold=0.01, n_clusters=5).fit(X)

X['Labels'] = brch.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], 
                palette=sns.color_palette('bright', np.unique(brch.labels_).shape[0]))
plt.title('Birch with ' + str(np.unique(brch.labels_).shape[0]) + ' Clusters')
plt.show()

# ###################### ALL-IN-ONE ######################

fig = plt.figure(figsize=(20,20))

a1 = fig.add_subplot(331)
a2 = fig.add_subplot(332)
a3 = fig.add_subplot(333)
a4 = fig.add_subplot(334)
a5 = fig.add_subplot(335)
a6 = fig.add_subplot(336)
a7 = fig.add_subplot(337)
a8 = fig.add_subplot(338)
a9 = fig.add_subplot(339)

##### KMeans-10 #####

km5 = KMeans(n_clusters=5).fit(X)
X['Labels'] = km5.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'],
                palette=sns.color_palette('bright', 5), s=60, ax=a1)
a1.set_title('KMeans - 5')

##### KMeans #####

km10 = KMeans(n_clusters=10).fit(X)
X['Labels'] = km10.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'],
                palette=sns.color_palette('bright', 10), s=60, ax=a2)
a9.set_title('KMeans - 10')

##### Agglomerative Clustering #####

agglom = AgglomerativeClustering(n_clusters=5, linkage='average').fit(X)
X['Labels'] = agglom.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'],
                palette=sns.color_palette('bright', 5), s=60, ax=a3)
a2.set_title('Agglomerative')


##### DBSCAN #####

db = DBSCAN(eps=50, min_samples=6).fit(X)
X['Labels'] = db.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(db.labels_).shape[0]), ax=a4)
a3.set_title('DBSCAN')


##### MEAN SHIFT #####

bandwidth = estimate_bandwidth(X, quantile=0.5)
ms = MeanShift(bandwidth).fit(X)
X['Labels'] = ms.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(ms.labels_).shape[0]), ax=a5)
a4.set_title('MeanShift')

##### Birch #####

brch = Birch(threshold=0.01, n_clusters=5).fit(X)
X['Labels'] = brch.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(brch.labels_).shape[0]), ax=a6)
a5.set_title('Birch')

##### Mini Batch K Means #####

mbk = MiniBatchKMeans(n_clusters=5).fit(X)
X['Labels'] = mbk.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(mbk.labels_).shape[0]), ax=a7)
a6.set_title('Mini-batch')

##### OPTICS #####

optics = OPTICS(eps=0.5, min_samples=20).fit(X)
X['Labels'] = optics.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(optics.labels_).shape[0]), ax=a8)
a7.set_title('OPTICS')

##### Affinity #####

affi = AffinityPropagation(damping=0.9).fit(X)
X['Labels'] = affi.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(affi.labels_).shape[0]), ax=a9)
a8.set_title('Affinity')

plt.tight_layout()
plt.show()

###################### Categorical Variables ######################

x=["Blue-white","Blue White","yellow-white","Blue white","Yellowish White","Blue-White","White-Yellow","Whitish","white"]
for i in x:
    strs.loc[strs["Color"]==i,"Color"]= "White"
    
for i in ["yellowish","Yellowish"]:
    strs.loc[strs["Color"]==i,"Color"]="Yellow"
    
for i in ["Orange-Red","Pale yellow orange"]:
    strs.loc[strs["Color"]==i,"Color"]="Orange"
    
###################### Data Manipulation and ML Comparison ######################
    
strs=pd.get_dummies(data=strs,columns=["Color","Spectral_Class"],drop_first=True)

Q=strs.drop("Type",axis=1)
y=strs["Type"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(Q,y,test_size=0.2,random_state=0)

from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)