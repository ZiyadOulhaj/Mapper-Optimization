import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import meshplot as mp
import gudhi                 as gd
from tqdm                    import tqdm
from sklearn.cluster import KMeans
from mapper import MapperComplex as ParallelMapperComplex
from scipy.stats import bernoulli
import itertools
from joblib import Parallel, delayed
from pyvis.network import Network
from SoftMapper import smooth_scheme, compute_mapper, filter_st
import time
from sklearn.decomposition import PCA
import scipy.sparse.csgraph as scs
import robust_laplacian as rlap
import scipy as sp

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
os.system('mkdir ' + 'results/' + name)
shape = sys.argv[2] #'Human/10.off'
n_clusters = int(sys.argv[3]) #3
resolutions = np.array([int(sys.argv[4])]) #25
gain = np.array([float(sys.argv[5])]) #0.3
sigma = float(sys.argv[6]) #0.01
lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=float(sys.argv[7]), decay_steps=int(sys.argv[8]), decay_rate=float(sys.argv[9])) #5e-2 10 0.01
n_epochs = int(sys.argv[10]) #200
K = int(sys.argv[11]) #10
mode = int(sys.argv[12]) #0

np.random.seed(0)

#shape = 'Octopus/132.off'
#n_clusters = 8
#resolutions = np.array([10])
#gain = np.array([0.3])
#sigma = 0.01
#lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=1e-0, decay_steps=10, decay_rate=.05)

#shape = 'Table/142.off'
#n_clusters = 8
#resolutions = np.array([10])
#gain = np.array([0.35])
#sigma = 0.01
#lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=1e-1, decay_steps=10, decay_rate=.1)

vertices, faces = off2numpy(path + shape)
kmeans = KMeans(n_clusters=n_clusters, n_init=10)

mp.offline()
dplot = mp.plot(vertices, faces, return_plot=True)
os.system('rm *.html')
dplot.save("results/" + name + "/3Dshape.html")

pca = PCA(n_components=2)
pca.fit(vertices)

X = tf.Variable(initial_value=vertices.astype(np.float32), trainable=False)

if mode == 0:
    dimension = 3
    params = tf.Variable(initial_value=np.ones([dimension,1]).astype(np.float32)/np.sqrt(dimension), trainable=True)
    f = tf.tensordot(X, params, axes=1)
elif mode == 1:
    dimension = 200
    laplacian, mass = rlap.mesh_laplacian(vertices, faces)
    _, egvecs = sp.sparse.linalg.eigsh(laplacian, dimension, mass, sigma=1e-8)
    params = tf.Variable(initial_value=np.ones([dimension,1]).astype(np.float32)/np.sqrt(dimension), trainable=True)
    egv = tf.Variable(initial_value=egvecs.astype(np.float32), trainable=False)
    f = tf.tensordot(egv, params, axes=1)
elif mode == 2:
    dimension = len(vertices)
    adjacency = get_adj_from_faces(faces, dimension)
    geodesics = scs.dijkstra(adjacency)
    gram = np.exp(-geodesics)
    G = tf.Variable(initial_value=gram.astype(np.float32), trainable=False)
    params = tf.Variable(initial_value=np.ones([dimension,1]).astype(np.float32)/np.sqrt(dimension), trainable=True)
    f = tf.tensordot(G, params, axes=1)

    #egvals, _ = sp.linalg.eigh(gram)
    #print(np.all(egvals >= 0))

mapper = ParallelMapperComplex(colors=f.numpy(), filters=f.numpy(), resolutions=resolutions, gains=gain, clustering=kmeans)
mapper.fit(vertices)
nt = mapper.get_pyvis()
nt.show("results/" + name + "/mapper_initial.html")

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)
losses = []
with Parallel(n_jobs=-1) as parallel:

    for epoch in tqdm(range(n_epochs+1)):

        if mode == 0:
            f = tf.tensordot(X, params, axes=1)
        elif mode == 1:
            f = tf.tensordot(egv, params, axes=1)
        elif mode == 2:
            f = tf.tensordot(G, params, axes=1)

        fn = (f - tf.math.reduce_min(f))/(tf.math.reduce_max(f) - tf.math.reduce_min(f))
        scheme = smooth_scheme(fn.numpy(), resolutions, gain, sigma)
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
            f_values = tf.repeat(f_values, clusters.shape[2], axis=2)

            filtration = tf.math.reduce_sum(f_values*clusters, axis=1)/np.sum(clusters, axis=1)
            l = parallel(delayed(filter_st)(list(st[k].get_skeleton(1)), filtration.numpy()[k]) for k in range(K))
            #l = parallel(delayed(filter_extended_st)(list(st[k].get_skeleton(1)), filtration.numpy()[k]) for k in range(K))
            loss = 0
            
            for k in range(K):
                dgm = tf.gather(filtration[k], l[k]) 
                loss = loss - tf.math.reduce_sum(tf.math.abs((dgm[:,1] - dgm[:,0])))/K

            regularization = tf.math.square(tf.norm(params) - 1)

            loss = loss + regularization

        losses.append(loss.numpy())
        gradients = tape.gradient(loss, [params])
        optimizer.apply_gradients(zip(gradients, [params]))

if mode == 0:
    print(params, tf.math.reduce_sum(params*np.array([[0],[0],[1]]))/tf.norm(params))

plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('results/' + name + '/loss')

mapper = ParallelMapperComplex(colors=f.numpy(), filters=f.numpy(), resolutions=resolutions, gains=gain, clustering=kmeans)
mapper.fit(vertices)
nt = mapper.get_pyvis()
nt.show("results/" + name + "/mapper_final.html")

plt.figure()
plt.hist(scheme.ravel())
plt.savefig('results/' + name + '/hist')

plt.figure()
order = np.argsort(fn.numpy(), axis=0)
plt.plot(fn.numpy()[order.ravel()], scheme[0,3,:][order], c='b');
plt.plot(fn.numpy()[order.ravel()], scheme[0,4,:][order], c='r');
plt.savefig('results/' + name + '/scheme')

