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

pd.set_option("display.max_columns", None)

# print(strs.head())

X = strs.drop(['Color', 'Spectral_Class'], axis=1)
sns.pairplot(data=X,hue="Type", palette=sns.color_palette('bright', 6))
plt.show()
sns.barplot(x="Type",y="Temperature",data=strs,
            palette=sns.color_palette('dark', 6))
plt.show()

###################### Correlation Map ######################

X_1 = strs.drop(['Type', 'Color', 'Spectral_Class'], axis=1)
# print(X_1.head())
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(X_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

###################### PCA and Normalization Application ######################

print(X)
norm = X.drop(['Type'], axis = 1)
print(norm.head())
from sklearn.preprocessing import normalize, StandardScaler
features = ['Temperature', 'Luminosity', 'Radius', 'Absolute Magnitude']
# Separating out the features
x = X.loc[:, features].values
# Separating out the target
y = X.loc[:,['Type']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
X_normalized = normalize(x)
X_normalized = pd.DataFrame(X_normalized)
X_normalized.columns = norm.columns

print(X_normalized.head())

# After standardization x array
print(x)

# Components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_normalized)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['pca1', 'pca2'])

# After applying pca
print(principalDf)
typear = X['Type']
typear = typear.astype(str)
typear.replace(['0', '1', '2', '3', '4', '5'],
               ['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants'], inplace=True)
print(typear)
principalDf.reset_index(drop=True, inplace=True)
typear.reset_index(drop=True, inplace=True)
finalDf = pd.concat([principalDf, typear], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('pca1', fontsize = 15)
ax.set_ylabel('pca2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants']
colors = ['red', 'green', 'blue', 'yellow', 'black', 'brown']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Type'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'pca1']
               , finalDf.loc[indicesToKeep, 'pca2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

print(finalDf)

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

# Let's see with 6 Clusters
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
plt.title('Agglomerative with ' +
          str(np.unique(agglom.labels_).shape[0]) + ' Clusters')
plt.show()

# ###################### Hierarchy dendogram ######################

from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 

dist = distance_matrix(X, X)

Z = hierarchy.linkage(dist, 'complete')
plt.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')

Z = hierarchy.linkage(dist, 'single')
plt.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

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
plt.title('MeanShift with ' + str(np.unique(ms.labels_).shape[0]) + ' Clusters' )
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

##### KMeans-5 #####

km5 = KMeans(n_clusters=5).fit(X)
X['Labels'] = km5.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'],
                palette=sns.color_palette('bright', 5), s=60, ax=a1)
a1.set_title('KMeans - 5')

##### KMeans-10 #####

km10 = KMeans(n_clusters=10).fit(X)
X['Labels'] = km10.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'],
                palette=sns.color_palette('bright', 10), s=60, ax=a2)
a2.set_title('KMeans - 10')

##### Agglomerative Clustering #####

agglom = AgglomerativeClustering(n_clusters=5, linkage='average').fit(X)
X['Labels'] = agglom.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'],
                palette=sns.color_palette('bright', 5), s=60, ax=a3)
a3.set_title('Agglomerative')


##### DBSCAN #####

db = DBSCAN(eps=50, min_samples=6).fit(X)
X['Labels'] = db.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(db.labels_).shape[0]), ax=a4)
a4.set_title('DBSCAN')


##### MEAN SHIFT #####

bandwidth = estimate_bandwidth(X, quantile=0.5)
ms = MeanShift(bandwidth).fit(X)
X['Labels'] = ms.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(ms.labels_).shape[0]), ax=a5)
a5.set_title('MeanShift')

##### Birch #####

brch = Birch(threshold=0.01, n_clusters=5).fit(X)
X['Labels'] = brch.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(brch.labels_).shape[0]), ax=a6)
a6.set_title('Birch')

##### Mini Batch K Means #####

mbk = MiniBatchKMeans(n_clusters=5).fit(X)
X['Labels'] = mbk.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(mbk.labels_).shape[0]), ax=a7)
a7.set_title('Mini-batch')

##### OPTICS #####

optics = OPTICS(eps=0.5, min_samples=20).fit(X)
X['Labels'] = optics.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(optics.labels_).shape[0]), ax=a8)
a8.set_title('OPTICS')

##### Affinity #####

affi = AffinityPropagation(damping=0.9).fit(X)
X['Labels'] = affi.labels_
sns.scatterplot(X['Temperature'], X['Absolute Magnitude'], hue=X['Labels'], style=X['Labels'], s=60,
                palette=sns.color_palette('bright', np.unique(affi.labels_).shape[0]), ax=a9)
a9.set_title('Affinity')

plt.tight_layout()
plt.show()

########## PCA applied data visualization under clustering algorithms ##########

###################### KMeans 3-5-10 PCA and Norm applied ######################

from sklearn.cluster import KMeans

# 3 cluster
km3 = KMeans(n_clusters=3).fit(X)

finalDf['Type'] = km3.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', 3))
plt.title('KMeans with 3 Clusters on PCA applied')
plt.show()

# Let's see with 6 Clusters
km6 = KMeans(n_clusters=6).fit(X)

finalDf['Type'] = km6.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', 6))
plt.title('KMeans with 6 Clusters on PCA applied')
plt.show()

# Let's see with 10 Clusters
km10 = KMeans(n_clusters=10).fit(X)

finalDf['Type'] = km10.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', 10))
plt.title('KMeans with 10 Clusters on PCA applied')
plt.show()

###################### OPTICS ######################

from sklearn.cluster import OPTICS 

optics = OPTICS(eps=0.5, min_samples=20).fit(X)

finalDf['Type'] = optics.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', np.unique(optics.labels_).shape[0]))
plt.title('OPTICS with ' + str(np.unique(optics.labels_).shape[0]) + ' Clusters on PCA applied')
plt.show()

###################### Affinity Propagation ######################

from sklearn.cluster import AffinityPropagation

affi = AffinityPropagation(damping=0.9).fit(X)

finalDf['Type'] = affi.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', np.unique(affi.labels_).shape[0]))
plt.title('Affinity Prop with '+ str(np.unique(affi.labels_).shape[0]) + ' Clusters on PCA applied')
plt.show()

###################### Agglomerative ######################

from sklearn.cluster import AgglomerativeClustering 

agglom = AgglomerativeClustering(n_clusters=5, linkage='average').fit(X)

finalDf['Type'] = agglom.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', 5))
plt.title('Agglomerative with ' +
          str(np.unique(agglom.labels_).shape[0]) + ' Clusters on PCA applied')
plt.show()

###################### Hierarchy dendogram ######################

from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 

dist = distance_matrix(finalDf, finalDf)

Z = hierarchy.linkage(dist, 'complete')

plt.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size=12, orientation='right')

Z = hierarchy.linkage(dist, 'average')
plt.figure(figsize=(18, 50))
dendro = hierarchy.dendrogram(Z, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

###################### DBSCAN ######################

from sklearn.cluster import DBSCAN 

db = DBSCAN(eps=50, min_samples=6).fit(X)

finalDf['Type'] = db.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', np.unique(db.labels_).shape[0]))
plt.title('DBSCAN with epsilon 50, min samples 6 on PCA applied')
plt.show()

###################### MeanShift ######################

from sklearn.cluster import MeanShift, estimate_bandwidth

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.5)
ms = MeanShift(bandwidth).fit(X)

finalDf['Type'] = ms.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', np.unique(ms.labels_).shape[0]))
plt.plot()
plt.title('MeanShift with ' + str(np.unique(ms.labels_).shape[0]) + ' Clusters on PCA applied' )
plt.show()

###################### Mini Batch K Means ######################

from sklearn.cluster import MiniBatchKMeans

mbk = MiniBatchKMeans(n_clusters=5).fit(X)

finalDf['Type'] = mbk.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', np.unique(mbk.labels_).shape[0]))
plt.title('Mini-batch with ' + str(np.unique(mbk.labels_).shape[0]) + ' clusters on PCA applied')
plt.show()

###################### Birch ######################

from sklearn.cluster import Birch

brch = Birch(threshold=0.01, n_clusters=5).fit(X)

finalDf['Type'] = brch.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=finalDf['Type'], 
                palette=sns.color_palette('bright', np.unique(brch.labels_).shape[0]))
plt.title('Birch with ' + str(np.unique(brch.labels_).shape[0]) + ' Clusters on PCA applied')
plt.show()

###################### ALL-IN-ONE PCA Data ######################

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

##### KMeans-5 #####

km5 = KMeans(n_clusters=5).fit(X)
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=km5.labels_, style=km5.labels_,
                palette=sns.color_palette('bright', 5), s=60, ax=a1)
a1.set_title('KMeans - 5')

##### KMeans-10 #####

km10 = KMeans(n_clusters=10).fit(finalDf)
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=km10.labels_, style=km10.labels_,
                palette=sns.color_palette('bright', 10), s=60, ax=a2)
a2.set_title('KMeans - 10')

##### Agglomerative Clustering #####

agglom = AgglomerativeClustering(n_clusters=5, linkage='average').fit(finalDf)
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=agglom.labels_, style=agglom.labels_,
                palette=sns.color_palette('bright', 5), s=60, ax=a3)
a3.set_title('Agglomerative')


##### DBSCAN #####

db = DBSCAN(eps=50, min_samples=6).fit(finalDf)
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=db.labels_, style=db.labels_, s=60,
                palette=sns.color_palette('bright', np.unique(db.labels_).shape[0]), ax=a4)
a4.set_title('DBSCAN')


##### MEAN SHIFT #####

bandwidth = estimate_bandwidth(finalDf, quantile=0.5)
ms = MeanShift(bandwidth).fit(finalDf)
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=ms.labels_, style=ms.labels_, s=60,
                palette=sns.color_palette('bright', np.unique(ms.labels_).shape[0]), ax=a5)
a5.set_title('MeanShift')

##### Birch #####

brch = Birch(threshold=0.01, n_clusters=5).fit(finalDf)
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=brch.labels_, style=brch.labels_, s=60,
                palette=sns.color_palette('bright', np.unique(brch.labels_).shape[0]), ax=a6)
a6.set_title('Birch')

##### Mini Batch K Means #####

mbk = MiniBatchKMeans(n_clusters=5).fit(finalDf)
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=mbk.labels_, style=mbk.labels_, s=60,
                palette=sns.color_palette('bright', np.unique(mbk.labels_).shape[0]), ax=a7)
a7.set_title('Mini-batch')

##### OPTICS #####

optics = OPTICS(eps=0.5, min_samples=20).fit(finalDf)
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=optics.labels_, style=optics.labels_, s=60,
                palette=sns.color_palette('bright', np.unique(optics.labels_).shape[0]), ax=a8)
a8.set_title('OPTICS')

##### Affinity #####

affi = AffinityPropagation(damping=0.9).fit(finalDf)
sns.scatterplot(finalDf['pca1'], finalDf['pca2'], hue=affi.labels_, style=affi.labels_, s=60,
                palette=sns.color_palette('bright', np.unique(affi.labels_).shape[0]), ax=a9)
a9.set_title('Affinity')

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
X_train,X_test,y_train,y_test=train_test_split(Q,y,
                                               test_size=0.2,random_state=0)

from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)