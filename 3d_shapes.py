import os
import sys
import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pickle as pck
import tensorflow as tf
import meshplot as mp
import gudhi as gd
import scipy.sparse.csgraph as scs
import robust_laplacian as rlap

from tqdm import tqdm
from time import time
from scipy.stats import bernoulli
from joblib import Parallel, delayed
from pyvis.network import Network
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from umap import UMAP

from mapper import MapperComplex as ParallelMapperComplex
from SoftMapper import smooth_scheme, compute_mapper, filter_st

def off2numpy(shape_name):
    with open(shape_name, 'r') as S:
        S.readline()
        num_vertices, num_faces, _ = [int(n) for n in S.readline().split(' ')]
        info = S.readlines()
    vertices = np.array([[float(coord) for coord in l.split(' ')] for l in info[0:num_vertices]])
    faces    = np.array([[int(coord) for coord in l.split(' ')[1:]] for l in info[num_vertices:]])
    return vertices, faces

def get_adj_from_faces(faces, num_vertices):
    NG = [[0 for _ in range(num_vertices)] for _ in range(num_vertices)]
    for face in faces:
        [i1, i2, i3] = face
        NG[i1][i2] = 1
        NG[i2][i1] = 1
        NG[i1][i3] = 1
        NG[i3][i1] = 1
        NG[i3][i2] = 1
        NG[i2][i3] = 1
    return np.array(NG)

path = "./3dshapes/"

name = sys.argv[1] #'human10'

shape = sys.argv[2] #'Human/10.off'
n_clusters = int(sys.argv[3]) #3
resolutions = np.array([int(i) for i in sys.argv[4].split('-')]) #25
gains = np.array([float(g) for g in sys.argv[5].split('-')]) #0.3
sigma = float(sys.argv[6]) #0.01
initial_learning_rate = float(sys.argv[7]) #5e-2
decay_steps = int(sys.argv[8]) #10
decay_rate = float(sys.argv[9]) #0.1
n_epochs = int(sys.argv[10]) #200
K = int(sys.argv[11]) #10
mode = int(sys.argv[12]) #0
num_filtrations = int(sys.argv[13]) #1
idx_filtration = int(sys.argv[14]) #0

params = {
'shape': shape,
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

np.random.seed(0)

vertices, faces = off2numpy(path + shape)
dimension = len(vertices)
adjacency = get_adj_from_faces(faces, dimension)
geodesics = scs.dijkstra(adjacency)

pca = PCA(n_components=2)
Xpca = pca.fit_transform(vertices)
Cpca = pairwise_distances(Xpca)

umap = UMAP(n_components=2)
Xumap = umap.fit_transform(vertices)
Cumap = pairwise_distances(Xumap)

tsne = TSNE(n_components=2)
Xtsne = tsne.fit_transform(vertices)
Ctsne = pairwise_distances(Xtsne)

plt.figure()
plt.scatter(Xpca[:,0], Xpca[:,1], s=1)
plt.show()

plt.figure()
plt.scatter(Xtsne[:,0], Xtsne[:,1], s=1)
plt.show()

plt.figure()
plt.scatter(Xumap[:,0], Xumap[:,1], s=1)
plt.show()

kmeans = KMeans(n_clusters=n_clusters, n_init=10)

mp.offline()
dplot = mp.plot(vertices, faces, return_plot=True)
os.system('rm *.html')
dplot.save("results/" + name + "/3Dshape.html")

X = tf.Variable(initial_value=vertices.astype(np.float32), trainable=False)

if mode == 0:
    dimension = 3
    params = tf.Variable(initial_value=np.ones([dimension,num_filtrations]).astype(np.float32)/np.sqrt(dimension), trainable=True)
    f = tf.tensordot(X, params, axes=1)
elif mode == 1:
    dimension = 200
    laplacian, mass = rlap.mesh_laplacian(vertices, faces)
    _, egvecs = sp.sparse.linalg.eigsh(laplacian, dimension, mass, sigma=1e-8)
    params = tf.Variable(initial_value=np.ones([dimension,num_filtrations]).astype(np.float32)/np.sqrt(dimension), trainable=True)
    egv = tf.Variable(initial_value=egvecs.astype(np.float32), trainable=False)
    f = tf.tensordot(egv, params, axes=1)
elif mode == 2:
    gram = np.exp(-geodesics)
    G = tf.Variable(initial_value=gram.astype(np.float32), trainable=False)
    params = tf.Variable(initial_value=np.ones([dimension,num_filtrations]).astype(np.float32)/np.sqrt(dimension), trainable=True)
    f = tf.tensordot(G, params, axes=1)

    #egvals, _ = sp.linalg.eigh(gram)
    #print(np.all(egvals >= 0))

C1 = geodesics #pairwise_distances(vertices)

mapperbase = ParallelMapperComplex(colors=np.hstack([f.numpy(), vertices, Xpca, Xtsne, Xumap]), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=kmeans)
mapperbase.fit(vertices)
nt = mapperbase.get_pyvis()
nt.show("results/" + name + "/mapper_initial.html")

Gbase, Afbase, Adbase, Apbase, Atbase, Aubase = mapperbase.get_networkx(dimension=3)
Cmapperbf = scs.dijkstra(Afbase.todense(), directed=False)
Cmapperbd = scs.dijkstra(Adbase.todense(), directed=False)
Cmapperbp = scs.dijkstra(Apbase.todense(), directed=False)
Cmapperbt = scs.dijkstra(Atbase.todense(), directed=False)
Cmapperbu = scs.dijkstra(Aubase.todense(), directed=False)
Tbase = np.zeros([len(C1), len(Cmapperbd)])
for k in Gbase.nodes():
    for idx_pt in mapperbase.node_info[k]["indices"]:
        Tbase[idx_pt, k] = 1

lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)
losses, times = [], []
with Parallel(n_jobs=-1) as parallel:

    for epoch in tqdm(range(n_epochs+1)):

        start = time()

        if mode == 0:
            f = tf.tensordot(X, params, axes=1)
        elif mode == 1:
            f = tf.tensordot(egv, params, axes=1)
        elif mode == 2:
            f = tf.tensordot(G, params, axes=1)

        fn = (f - tf.math.reduce_min(f))/(tf.math.reduce_max(f) - tf.math.reduce_min(f))
        scheme = smooth_scheme(fn.numpy(), resolutions, gains, sigma)
        upscheme = np.repeat(scheme, K, axis=0)
        assignments = bernoulli.rvs(upscheme)
        st, clusters = compute_mapper(X.numpy(), kmeans, assignments, "point cloud", n_clusters)
        
        with tf.GradientTape() as tape:

            if mode == 0:
                f = tf.tensordot(X, params, axes=1)
            elif mode == 1:
                f = tf.tensordot(egv, params, axes=1)
            elif mode == 2:
                f = tf.tensordot(G, params, axes=1)
 
            f_values = tf.repeat(tf.expand_dims(f, axis=0), clusters.shape[0], axis=0)
            f_values = tf.repeat(f_values[:,:,idx_filtration:idx_filtration+1], clusters.shape[2], axis=2)
            clus_sums = np.sum(clusters,axis=1)
            clus_sums = np.where(clus_sums == 0, np.ones(clus_sums.shape), clus_sums)
            filtration = tf.math.reduce_sum(f_values*clusters, axis=1)/clus_sums
            l = parallel(delayed(filter_st)(list(st[k].get_skeleton(1)), filtration.numpy()[k]) for k in range(K))
            loss = 0
            
            for k in range(K):
                dgm = tf.gather(filtration[k], l[k]) 
                loss = loss - tf.math.reduce_sum(tf.math.abs((dgm[:,1] - dgm[:,0])))/K

            regularization = tf.math.square(tf.norm(params) - 1)
            loss = loss + regularization
    
        end = time()

        times.append(end-start)
        losses.append(loss.numpy())
        gradients = tape.gradient(loss, [params])
        optimizer.apply_gradients(zip(gradients, [params]))

plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('results/' + name + '/loss')

mapper = ParallelMapperComplex(colors=np.hstack([f.numpy(), vertices, Xpca, Xtsne, Xumap]), filters=f.numpy(), resolutions=resolutions, gains=gains, clustering=kmeans)
mapper.fit(vertices)
nt = mapper.get_pyvis()
nt.show("results/" + name + "/mapper_final.html")

G, Af, Ad, Ap, At, Au = mapper.get_networkx(dimension=3)
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

plt.figure()
plt.hist(scheme.ravel())
plt.savefig('results/' + name + '/hist')

plt.figure()
order = np.argsort(fn.numpy(), axis=0)
plt.plot(fn.numpy()[order.ravel()], scheme[0,3,:][order], c='b');
plt.plot(fn.numpy()[order.ravel()], scheme[0,4,:][order], c='r');
plt.savefig('results/' + name + '/scheme')

corrf = tf.math.reduce_sum(params.numpy()[:,0:1]*np.array([[0],[0],[1]]))/tf.norm(params)
print(corrf.numpy())

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
'corrf': corrf.numpy(),
}

pck.dump(results, open('results/' + name + '/results.pkl', 'wb'))

