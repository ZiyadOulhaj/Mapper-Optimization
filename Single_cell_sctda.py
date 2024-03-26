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

# Import data

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

# Preprocessed data using Seurat in R

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

# Linear filter optimization

np.random.seed(1)
subset = np.random.randint(low=0, high=len(timepoints), size=500)
delta = np.max([directed_hausdorff(X, X[subset,:]), directed_hausdorff(X[subset,:], X)])
ag = AgglomerativeClustering(n_clusters=None, distance_threshold=8e-2*delta)
params = tf.Variable(initial_value=np.ones((p,1)).astype(np.float32)/np.sqrt(p), trainable=True)
lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

f = tf.tensordot(X.astype(np.float32), params, axes=1)
mapperbase = MapperComplex(colors=np.hstack([np.array(timepoints).reshape((-1,1)), X, Xpca, Xtsne, Xumap]), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=ag)
mapperbase.fit(X)
nt = mapperbase.get_pyvis()
nt.show('sctda_initial.html')

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

dff = pd.DataFrame(list(zip(timepoints, list(f.numpy().ravel()))), columns=['time', 'filter'])
ax = sns.displot(dff, x="filter", hue="time", kind="kde", palette=cm.viridis)
ax.set(xlabel=None, ylabel=None)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 0), ncol=5, title=None, frameon=False)
plt.savefig('sctda_densities_initial.png')

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

plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

tf.norm(params)

mapper = MapperComplex(colors=np.hstack([np.array(timepoints).reshape((-1,1)), X, Xpca, Xtsne, Xumap]), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=ag)
mapper.fit(X)
nt = mapper.get_pyvis()
nt.show('sctda_final.html')

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

cost_pca, cost_tsne, cost_umap = 0., 0., 0. 
cost_mapperbf, cost_mapperf = 0., 0.
cost_mapperbd, cost_mapperd = 0., 0.
cost_mapperbp, cost_mapperp = 0., 0.
cost_mapperbt, cost_mappert = 0., 0.
cost_mapperbu, cost_mapperu = 0., 0.

for i in range(0,C1.shape[0],100):
#    print(str(i) + '/' + str(len(C1)))
    for j in range(0,C1.shape[1],100):
#        print(str(j) + '/' + str(len(C1)))
        Cmapperbf_cost  = np.where(np.isinf(Cmapperbf), C1[i,j]*np.ones(Cmapperbf.shape), Cmapperbf)
        Cmapperbd_cost  = np.where(np.isinf(Cmapperbd), C1[i,j]*np.ones(Cmapperbd.shape), Cmapperbd)
        Cmapperbp_cost  = np.where(np.isinf(Cmapperbp), C1[i,j]*np.ones(Cmapperbp.shape), Cmapperbp)
        Cmapperbt_cost  = np.where(np.isinf(Cmapperbt), C1[i,j]*np.ones(Cmapperbt.shape), Cmapperbt)
        Cmapperbu_cost  = np.where(np.isinf(Cmapperbu), C1[i,j]*np.ones(Cmapperbu.shape), Cmapperbu)

        Cmapperf_cost   = np.where(np.isinf(Cmapperf),  C1[i,j]*np.ones(Cmapperf.shape),  Cmapperf)
        Cmapperd_cost   = np.where(np.isinf(Cmapperd),  C1[i,j]*np.ones(Cmapperd.shape),  Cmapperd)
        Cmapperp_cost   = np.where(np.isinf(Cmapperp),  C1[i,j]*np.ones(Cmapperp.shape),  Cmapperp)
        Cmappert_cost   = np.where(np.isinf(Cmappert),  C1[i,j]*np.ones(Cmappert.shape),  Cmappert)
        Cmapperu_cost   = np.where(np.isinf(Cmapperu),  C1[i,j]*np.ones(Cmapperu.shape),  Cmapperu)

        cost_pca        = max(cost_pca,        np.abs(Cpca[i,j]  - C1[i,j]))
        cost_tsne       = max(cost_tsne,       np.abs(Ctsne[i,j] - C1[i,j]))
        cost_umap       = max(cost_umap,       np.abs(Cumap[i,j] - C1[i,j]))

        cost_mapperbf   = max(cost_mapperbf,   np.multiply( (Tbase[i:i+1,:].T).dot(Tbase[j:j+1,:]), np.abs(Cmapperbf_cost - C1[i,j]) ).max())
        cost_mapperbd   = max(cost_mapperbd,   np.multiply( (Tbase[i:i+1,:].T).dot(Tbase[j:j+1,:]), np.abs(Cmapperbd_cost - C1[i,j]) ).max())
        cost_mapperbp   = max(cost_mapperbp,   np.multiply( (Tbase[i:i+1,:].T).dot(Tbase[j:j+1,:]), np.abs(Cmapperbp_cost - C1[i,j]) ).max())
        cost_mapperbt   = max(cost_mapperbt,   np.multiply( (Tbase[i:i+1,:].T).dot(Tbase[j:j+1,:]), np.abs(Cmapperbt_cost - C1[i,j]) ).max())
        cost_mapperbu   = max(cost_mapperbu,   np.multiply( (Tbase[i:i+1,:].T).dot(Tbase[j:j+1,:]), np.abs(Cmapperbu_cost - C1[i,j]) ).max())

        cost_mapperf   = max(cost_mapperf,   np.multiply( (T[i:i+1,:].T).dot(T[j:j+1,:]), np.abs(Cmapperf_cost - C1[i,j]) ).max())
        cost_mapperd   = max(cost_mapperd,   np.multiply( (T[i:i+1,:].T).dot(T[j:j+1,:]), np.abs(Cmapperd_cost - C1[i,j]) ).max())
        cost_mapperp   = max(cost_mapperp,   np.multiply( (T[i:i+1,:].T).dot(T[j:j+1,:]), np.abs(Cmapperp_cost - C1[i,j]) ).max())
        cost_mappert   = max(cost_mappert,   np.multiply( (T[i:i+1,:].T).dot(T[j:j+1,:]), np.abs(Cmappert_cost - C1[i,j]) ).max())
        cost_mapperu   = max(cost_mapperu,   np.multiply( (T[i:i+1,:].T).dot(T[j:j+1,:]), np.abs(Cmapperu_cost - C1[i,j]) ).max())

print(cost_mapperbf, cost_mapperf)
print(cost_mapperbd, cost_mapperd)
print(cost_pca,  cost_mapperbp, cost_mapperp)
print(cost_tsne, cost_mapperbt, cost_mappert)
print(cost_umap, cost_mapperbu, cost_mapperu)

dff = pd.DataFrame(list(zip(timepoints, list(f.numpy().ravel()))), columns =['time', 'filter'])
ax = sns.displot(dff, x="filter", hue="time", kind="kde", palette=cm.viridis)
ax.set(xlabel=None, ylabel=None)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 0), ncol=5, title=None, frameon=False)
plt.savefig('sctda_densities_final.png')

initialparams = tf.Variable(initial_value=np.ones((p,1)).astype(np.float32)/np.sqrt(p), trainable=True)
fi = tf.tensordot(X.astype(np.float32), initialparams, axes=1)

corrfi,    corrf     = pearsonr(fi.numpy()[:,0], timepoints), pearsonr(f.numpy()[:,0], timepoints)
corrpca0,  corrpca1  = pearsonr(Xpca[:,0],       timepoints), pearsonr(Xpca[:,1],      timepoints)
corrtsne0, corrtsne1 = pearsonr(Xtsne[:,0],      timepoints), pearsonr(Xtsne[:,1],     timepoints)
corrumap0, corrumap1 = pearsonr(Xumap[:,0],      timepoints), pearsonr(Xumap[:,1],     timepoints)

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
'cost_pca': cost_pca,
'cost_tsne': cost_tsne,
'cost_umap': cost_umap,
'cost_mapperbf': cost_mapperbf,
'cost_mapperbd': cost_mapperbd,
'cost_mapperbp': cost_mapperbp,
'cost_mapperbt': cost_mapperbt,
'cost_mapperbu': cost_mapperbu,
'cost_mapperf': cost_mapperf,
'cost_mapperd': cost_mapperd,
'cost_mapperp': cost_mapperp,
'cost_mappert': cost_mappert,
'cost_mapperu': cost_mapperu,
'corr_mapper': [corrfi, corrf],
'corr_pca': [corrpca0, corrpca1],
'corr_tsne': [corrtsne0, corrtsne1],
'corr_umap': [corrumap0, corrumap1],
}

pck.dump(results, open('results/' + name + '/results.pkl', 'wb'))

mapper = MapperComplex(colors=np.array(dfn['HTR3E']).reshape((-1,1)), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=ag)
mapper.fit(X)
nt = mapper.get_pyvis(cmap=cm.hot)
nt.show('sctda_gene1.html')

mapper = MapperComplex(colors=np.array(dfn['CDX1']).reshape((-1,1)), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=ag)
mapper.fit(X)
nt = mapper.get_pyvis(cmap=cm.hot)
nt.show('sctda_gene2.html')
