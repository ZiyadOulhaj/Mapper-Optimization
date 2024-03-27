import os
import sys
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy as sp
import pickle as pck
import pandas as pd
import tensorflow as tf
import meshplot as mp
import gudhi as gd
import scipy.sparse.csgraph as scs
import seaborn as sns
import robust_laplacian as rlap

from tqdm import tqdm
from time import time
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import bernoulli, pearsonr
from joblib import Parallel, delayed
from pyvis.network import Network
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from umap import UMAP

from mapper import MapperComplex
from SoftMapper import smooth_scheme, compute_mapper, filter_extended_st

name = sys.argv[1] #'sctda'

path = sys.argv[2] #'Datasets/scTDA/'
n_clusters = int(sys.argv[3]) #3
resolutions = np.array([int(i) for i in sys.argv[4].split('-')]) #25
gains = np.array([float(g) for g in sys.argv[5].split('-')]) #0.3
sigma = float(sys.argv[6]) #1e-5
initial_learning_rate = float(sys.argv[7]) #5e-4
decay_steps = int(sys.argv[8]) #10
decay_rate = float(sys.argv[9]) #1
n_epochs = int(sys.argv[10]) #200
K = int(sys.argv[11]) #10
mode = int(sys.argv[12]) #0
num_filtrations = int(sys.argv[13]) #1
idx_filtration = int(sys.argv[14]) #0

params = {
'shape': path,
'n_clusters': n_clusters,
'resolutions': resolutions,
'gains': gains,
'sigma': sigma,
'initial_learning_rate': initial_learning_rate,
'decay_steps': decay_steps,
'decay_rate': decay_rate,
'n_epochs': n_epochs,
'K': K,
'mode': mode,
'num_filtrations': num_filtrations,
'idx_filtration': idx_filtration,
}

os.system('mkdir ' + 'results/' + name)
pck.dump(params, open('results/' + name + '/params.pkl', 'wb'))

files = []
cells = []
libs = []
days = []
with open(path + 'data.txt', 'r') as f:
    for line in f:
        sp = line[:-1].split('\t')
        files.append(sp[0])
        cells.append(int(sp[1]))
        libs.append(sp[2])
        days.append(int(sp[3]))

timepoints, dfs = [], []
for i in range(len(files)):
    f = files[i]
    dfs.append(pd.read_csv(path + f, sep="\t", header=None, index_col=0))
    timepoints = timepoints + [days[i]]*dfs[i].shape[1]
df = (pd.concat(dfs,axis=1)).transpose()

dfn = pd.read_csv(r"Datasets/seurat_normalized.csv",index_col=0).transpose()
X = np.array(dfn)
(n,p) = X.shape

C1 = pairwise_distances(X)

pca = PCA(n_components=2)
Xpca = pca.fit_transform(X)
Cpca = pairwise_distances(Xpca)

umap = UMAP(n_components=2)
Xumap = umap.fit_transform(X)
Cumap = pairwise_distances(Xumap)

tsne = TSNE(n_components=2)
Xtsne = tsne.fit_transform(X)
Ctsne = pairwise_distances(Xtsne)

np.random.seed(1)
subset = np.random.randint(low=0, high=len(timepoints), size=500)
delta = np.max([directed_hausdorff(X, X[subset,:]), directed_hausdorff(X[subset,:], X)])
ag = AgglomerativeClustering(n_clusters=None, distance_threshold=8e-2*delta)
params = tf.Variable(initial_value=np.ones((p,1)).astype(np.float32)/np.sqrt(p), trainable=True)
lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

f = tf.tensordot(X.astype(np.float32), params, axes=1)
f_base = f.numpy().ravel()

mapperbase = MapperComplex(colors=np.hstack([np.array(timepoints).reshape((-1,1)), X, Xpca, Xtsne, Xumap]), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=ag)
mapperbase.fit(X)

Gbase, Afbase, Adbase, Apbase, Atbase, Aubase = mapperbase.get_networkx(dimension=p)
Cmapperbf = scs.dijkstra(Afbase.todense(), directed=False)
Cmapperbd = scs.dijkstra(Adbase.todense(), directed=False)
Cmapperbp = scs.dijkstra(Apbase.todense(), directed=False)
Cmapperbt = scs.dijkstra(Atbase.todense(), directed=False)
Cmapperbu = scs.dijkstra(Aubase.todense(), directed=False)
Tbase = np.zeros([len(C1), len(Cmapperbd)])
for k in Gbase.nodes():
    for idx_pt in mapperbase.node_info[k]["indices"]:
        Tbase[idx_pt, k] = 1

losses, times = [], []
with Parallel(n_jobs=-1) as parallel:

    for epoch in tqdm(range(n_epochs+1)):

        start = time()

        f = tf.tensordot(X.astype(np.float32),params,axes=1)
        fn = (f - tf.math.reduce_min(f))/(tf.math.reduce_max(f) - tf.math.reduce_min(f))
        scheme = smooth_scheme(fn.numpy(), resolutions, gains, sigma)
        upscheme = np.repeat(scheme, K, axis=0)
        assignments = bernoulli.rvs(upscheme, random_state=0)
        st, clusters = compute_mapper(X, ag, assignments, maximum=5)
        
        with tf.GradientTape() as tape:
            f = tf.tensordot(X.astype(np.float32), params, axes=1)
            f_values = tf.repeat(tf.expand_dims(f, axis=0), clusters.shape[0], axis=0)
            f_values = tf.repeat(f_values, clusters.shape[2], axis=2)
            filtration = tf.math.reduce_sum(f_values*clusters, axis=1)/(np.sum(clusters, axis=1) + 1e-10)
            l = parallel(delayed(filter_extended_st)(list(st[k].get_skeleton(1)), filtration.numpy()[k]) for k in range(K))
            loss = 0
            
            for k in range(K):
                dgm = tf.gather(filtration[k],l[k]) 
                loss = loss - tf.math.reduce_sum(tf.math.abs((dgm[:,1]-dgm[:,0])))/K

            regularization = tf.math.square(tf.norm(params) - 1)
            loss = loss + regularization
    
        end = time()

        times.append(end-start)
        gradients = tape.gradient(loss, [params])
        optimizer.apply_gradients(zip(gradients, [params]))
        losses.append(loss.numpy())

f_final = f.numpy().ravel()

mapper = MapperComplex(colors=np.hstack([np.array(timepoints).reshape((-1,1)), X, Xpca, Xtsne, Xumap]), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=ag)
mapper.fit(X)

G, Af, Ad, Ap, At, Au = mapper.get_networkx(dimension=p)
Cmapperf = scs.dijkstra(Af.todense(), directed=False)
Cmapperd = scs.dijkstra(Ad.todense(), directed=False)
Cmapperp = scs.dijkstra(Ap.todense(), directed=False)
Cmappert = scs.dijkstra(At.todense(), directed=False)
Cmapperu = scs.dijkstra(Au.todense(), directed=False)
T = np.zeros([len(C1), len(Cmapperd)])
for k in G.nodes():
    for idx_pt in mapper.node_info[k]["indices"]:
        T[idx_pt, k] = 1

matrices = [Cmapperbf, Cmapperbd, Cmapperbp, Cmapperbt, Cmapperbu, Cmapperf, Cmapperd, Cmapperp, Cmappert, Cmapperu]

costs_baseline = [0., 0., 0.]
costs = [0. for _ in range(len(matrices))]

for i in range(0,C1.shape[0],100):
    for j in range(0,C1.shape[1],100):
        costs_baseline[0] = max(costs_baseline[0], np.abs(Cpca[i,j]  - C1[i,j]))
        costs_baseline[1] = max(costs_baseline[1], np.abs(Ctsne[i,j] - C1[i,j]))
        costs_baseline[2] = max(costs_baseline[2], np.abs(Cumap[i,j] - C1[i,j]))    
        for idx_m, matrix in enumerate(matrices):
            matrix_cost = np.where(np.isinf(matrix), C1[i,j]*np.ones(matrix.shape), matrix)
            if idx_m <= 4:
                costs[idx_m] = max(costs[idx_m], np.multiply( (Tbase[i:i+1,:].T).dot(Tbase[j:j+1,:]), np.abs(matrix_cost - C1[i,j]) ).max())
            else:
                costs[idx_m] = max(costs[idx_m], np.multiply( (T[i:i+1,:].T).dot(T[j:j+1,:]), np.abs(matrix_cost - C1[i,j]) ).max())

print(costs[0], costs[5])
print(costs[1], costs[6])
print(costs_baseline[0], costs[2], costs[7])
print(costs_baseline[1], costs[3], costs[8])
print(costs_baseline[2], costs[4], costs[9])

scores_baseline = [0., 0., 0.]
scores = [0. for _ in range(len(matrices))]

clustering_baseline = AgglomerativeClustering(n_clusters=len(np.unique(timepoints)), linkage='single')

for idx_b, reduced_data in enumerate([Xpca, Xtsne, Xumap]):
    clustering_baseline.fit(reduced_data)
    scores_baseline[idx_b] = adjusted_rand_score(timepoints, clustering_baseline.labels_)

clustering_mapper = AgglomerativeClustering(n_clusters=len(np.unique()), metric='precomputed', linkage='single')

clus_labels = np.ones([len(vertices)])
for idx_m, matrix in enumerate(matrices):
    clustering_mapper.fit(matrix)
    if idx_m <= 4:
        for k in Gbase.nodes():
            for idx_pt in mapperbase.node_info[k]["indices"]:
                clus_labels[idx_pt] = clustering_mapper.labels_[k]
    else:
        for k in G.nodes():
            for idx_pt in mapper.node_info[k]["indices"]:
                clus_labels[idx_pt] = clustering_mapper.labels_[k]
    scores[idx_m] = adjusted_rand_score(timepoints, clus_labels)

print(scores[0], scores[5])
print(scores[1], scores[6])
print(scores_baseline[0], scores[2], scores[7])
print(scores_baseline[1], scores[3], scores[8])
print(scores_baseline[2], scores[4], scores[9])

corrfi,    corrf     = pearsonr(f_base,     timepoints), pearsonr(f_final,    timepoints)
corrpca0,  corrpca1  = pearsonr(Xpca[:,0],  timepoints), pearsonr(Xpca[:,1],  timepoints)
corrtsne0, corrtsne1 = pearsonr(Xtsne[:,0], timepoints), pearsonr(Xtsne[:,1], timepoints)
corrumap0, corrumap1 = pearsonr(Xumap[:,0], timepoints), pearsonr(Xumap[:,1], timepoints)

print(corrfi,    corrf)
print(corrpca0,  corrpca1)
print(corrtsne0, corrtsne1)
print(corrumap0, corrumap1)

results = {
'filter': params.numpy(),
'loss': losses,
'time': times,
'scheme': scheme,
'pca': Xpca,
'tsne': Xtsne,
'umap': Xumap,
'costs_baseline': costs_baseline,
'costs': costs,
'scores_baseline': scores_baseline,
'scores': scores,
'corr_mapper': [corrfi, corrf],
'corr_pca': [corrpca0, corrpca1],
'corr_tsne': [corrtsne0, corrtsne1],
'corr_umap': [corrumap0, corrumap1],
}

pck.dump(results, open('results/' + name + '/results.pkl', 'wb'))

plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('results/' + name + '/losses')

plt.figure()
plt.boxplot(times)
plt.savefig('results/' + name + '/times')

plt.figure()
plt.scatter(Xpca[:,0], Xpca[:,1], s=1)
plt.savefig('results/' + name + '/pca')

plt.figure()
plt.scatter(Xtsne[:,0], Xtsne[:,1], s=1)
plt.savefig('results/' + name + '/tsne')

plt.figure()
plt.scatter(Xumap[:,0], Xumap[:,1], s=1)
plt.savefig('results/' + name + '/umap')

dff = pd.DataFrame(list(zip(timepoints, list(f_base))), columns=['time', 'filter'])
ax = sns.displot(dff, x="filter", hue="time", kind="kde", palette=cm.viridis)
ax.set(xlabel=None, ylabel=None)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 0), ncol=5, title=None, frameon=False)
plt.savefig("results/" + name + "/sctda_densities_initial.png")

dff = pd.DataFrame(list(zip(timepoints, list(f_final))), columns =['time', 'filter'])
ax = sns.displot(dff, x="filter", hue="time", kind="kde", palette=cm.viridis)
ax.set(xlabel=None, ylabel=None)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 0), ncol=5, title=None, frameon=False)
plt.savefig("results/" + name + "/sctda_densities_final.png")

nt = mapperbase.get_pyvis()
nt.show("results/" + name + "/mapper_initial_time.html")

nt = mapper.get_pyvis()
nt.show("results/" + name + "/mapper_final_time.html")

mapper_htr3e = MapperComplex(colors=np.array(dfn['HTR3E']).reshape((-1,1)), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=ag)
mapper_htr3e.fit(X)
nt = mapper_htr3e.get_pyvis(cmap=cm.hot)
nt.show("results/" + name + "/mapper_final_htr3e.html")

mapper_cdx1 = MapperComplex(colors=np.array(dfn['CDX1']).reshape((-1,1)), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=ag)
mapper_cdx1.fit(X)
nt = mapper_cdx1.get_pyvis(cmap=cm.hot)
nt.show("results/" + name + "/mapper_final_cdx1.html")

