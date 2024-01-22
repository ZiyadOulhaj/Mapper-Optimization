#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import meshplot as mp
import gudhi                 as gd
from tqdm                    import tqdm
from sklearn.cluster import KMeans
from mapper import MapperComplex
from scipy.stats import bernoulli
import itertools
from joblib import Parallel, delayed
from pyvis.network import Network
from SoftMapper import smooth_scheme, compute_mapper, filter_st
import time
from sklearn.decomposition import PCA


# ## Utils for importing shapes

# In[3]:


def off2numpy(shape_name):
    with open(shape_name, 'r') as S:
        S.readline()
        num_vertices, num_faces, _ = [int(n) for n in S.readline().split(' ')]
        info = S.readlines()
    vertices = np.array([[float(coord) for coord in l.split(' ')] for l in info[0:num_vertices]])
    faces    = np.array([[int(coord) for coord in l.split(' ')[1:]] for l in info[num_vertices:]])
    return vertices, faces


# ## Human

# In[4]:


#path=r"/Users/ziyad/Desktop/LabeledDB_new/"
path=r"/home/mcarrier/Seafile/[Docs] Talks/gudhi/data/3dshapes/"
vertices, faces = off2numpy(path + 'Human/10.off')
mp.plot(vertices,faces);


# In[5]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(vertices)
pca.components_


# In[6]:


n_clusters=3
resolutions=np.array([25])
gain=np.array([0.3])

sigma=0.01

kmeans=KMeans(n_clusters=n_clusters)


params=tf.Variable(initial_value=np.array([[1],[1],[1]]).astype(np.float32)/np.sqrt(3),trainable=True)
X=tf.Variable(initial_value=vertices.astype(np.float32), trainable=False)

lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=5e-2, decay_steps=10, decay_rate=.01)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)


# In[7]:


f=tf.tensordot(X,params,axes=1)
mapper=MapperComplex(colors=f.numpy(),filters=f.numpy(),resolutions=resolutions, gains=gain, clustering=kmeans)
mapper.fit(vertices)
nt=mapper.get_pyvis()
nt.show('human_initial.html')


# In[8]:


losses = []
K=10
with Parallel(n_jobs=-1) as parallel:
    for epoch in tqdm(range(200+1)):

        f=tf.tensordot(X,params,axes=1)
        fn=(f-tf.math.reduce_min(f))/(tf.math.reduce_max(f)-tf.math.reduce_min(f))
        scheme=smooth_scheme(fn.numpy(),resolutions,gain,sigma)

        upscheme=np.repeat(scheme,K,axis=0)

        assignments=bernoulli.rvs(upscheme)

        st,clusters=compute_mapper(X.numpy(),kmeans,assignments)
        
        with tf.GradientTape() as tape:
            f=tf.tensordot(X,params,axes=1)
            f_values=tf.repeat(tf.expand_dims(f,axis=0),clusters.shape[0],axis=0)

            f_values=tf.repeat(f_values,clusters.shape[2],axis=2)

            filtration=tf.math.reduce_sum(f_values*clusters,axis=1)/np.sum(clusters,axis=1)
            #l=[(_LowerStarSimplexTree(st[k],filtration[k],[0],11,False)[0][0]).reshape(-1,2) for k in range(n)]
            l=parallel(delayed(filter_st)(list(st[k].get_skeleton(1)),filtration.numpy()[k]) for k in range(K))
            l=parallel(delayed(filter_extended_st)(list(st[k].get_skeleton(1)),filtration.numpy()[k]) for k in range(K))
            loss=0
            
            #dgm=tf.concat([tf.gather(filtration[k],l[k]) for k in range(K)],axis=0)
            for k in range(K):
                dgm=tf.gather(filtration[k],l[k]) 
                loss=loss-tf.math.reduce_sum(tf.math.abs((dgm[:,1]-dgm[:,0])))/K

            regularization = tf.math.square(tf.norm(params)-1)

            loss=loss+regularization

        gradients = tape.gradient(loss, [params])
        
        optimizer.apply_gradients(zip(gradients, [params]))


        losses.append(loss.numpy())


# In[9]:


plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[10]:


params


# In[11]:


tf.norm(params)


# In[12]:


tf.math.reduce_sum(params*np.array([[0],[1],[0]]))/tf.norm(params)


# In[13]:


mapper=MapperComplex(colors=f.numpy(),filters=f.numpy(),resolutions=resolutions, gains=gain, clustering=kmeans)
mapper.fit(vertices)
nt=mapper.get_pyvis()
nt.show('human_final.html')


# In[14]:


plt.hist(scheme.ravel())


# In[15]:


order=np.argsort(fn.numpy(),axis=0)


# In[16]:


plt.plot(fn.numpy()[order.ravel()],scheme[0,3,:][order],c='b');
plt.plot(fn.numpy()[order.ravel()],scheme[0,4,:][order],c='r');


# In[17]:


plt.plot(fn.numpy()[order.ravel()],scheme[0,3,:][order],c='b');
plt.plot(fn.numpy()[order.ravel()],scheme[0,4,:][order],c='r');
plt.xlim(0.05, 0.25)


# ## Octopus

# In[98]:


#path=r"/Users/ziyad/Desktop/LabeledDB_new/"
vertices, faces = off2numpy(path + 'Octopus/132.off')
mp.plot(vertices,faces);


# In[99]:


pca = PCA(n_components=2)
pca.fit(vertices)
pca.components_


# In[66]:


n_clusters=8
resolutions=np.array([10])
gain=np.array([0.3])
sigma=0.01

kmeans=KMeans(n_clusters=n_clusters)

params=tf.Variable(initial_value=np.array([[1],[1],[1]]).astype(np.float32)/np.sqrt(3),trainable=True)
X=tf.Variable(initial_value=vertices.astype(np.float32), trainable=False)

lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=1e-0, decay_steps=10, decay_rate=.05)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)


# In[67]:


f=tf.tensordot(X,params,axes=1)
mapper=MapperComplex(colors=f.numpy(),filters=f.numpy(),resolutions=resolutions, gains=gain, clustering=kmeans)
mapper.fit(vertices)
nt=mapper.get_pyvis()
nt.show('octopus_initial.html')


# In[68]:


losses = []
K=10
with Parallel(n_jobs=-1) as parallel:
    for epoch in tqdm(range(200+1)):

        f=tf.tensordot(X,params,axes=1)
        fn=(f-tf.math.reduce_min(f))/(tf.math.reduce_max(f)-tf.math.reduce_min(f))
        scheme=smooth_scheme(fn.numpy(),resolutions,gain,sigma)

        upscheme=np.repeat(scheme,K,axis=0)

        assignments=bernoulli.rvs(upscheme)

        st,clusters=compute_mapper(X.numpy(),kmeans,assignments,maximum=8)
        
        with tf.GradientTape() as tape:
            f=tf.tensordot(X,params,axes=1)
            f_values=tf.repeat(tf.expand_dims(f,axis=0),clusters.shape[0],axis=0)

            f_values=tf.repeat(f_values,clusters.shape[2],axis=2)

            filtration=tf.math.reduce_sum(f_values*clusters,axis=1)/np.sum(clusters,axis=1)
            #l=[(_LowerStarSimplexTree(st[k],filtration[k],[0],11,False)[0][0]).reshape(-1,2) for k in range(n)]
            l=parallel(delayed(filter_st)(list(st[k].get_skeleton(1)),filtration.numpy()[k]) 
                for k in range(K))
            loss=0
            
            #dgm=tf.concat([tf.gather(filtration[k],l[k]) for k in range(K)],axis=0)
            for k in range(K):
                dgm=tf.gather(filtration[k],l[k]) 
                loss=loss-tf.math.reduce_sum(tf.math.abs((dgm[:,1]-dgm[:,0])))/K

            regularization = tf.math.square(tf.norm(params)-1)

            loss=loss+regularization

        gradients = tape.gradient(loss, [params])
        
        optimizer.apply_gradients(zip(gradients, [params]))


        losses.append(loss.numpy())


# In[73]:


plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[70]:


params


# In[71]:


tf.norm(params)


# In[75]:


tf.math.reduce_sum(params*np.array([[0],[0],[1]]))/tf.norm(params)


# In[72]:


mapper=MapperComplex(colors=f.numpy(),filters=f.numpy(),resolutions=resolutions, gains=gain, clustering=kmeans)
mapper.fit(vertices)
nt=mapper.get_pyvis()
nt.show('octopus_final.html')


# ## Table

# In[24]:


path=r"/Users/ziyad/Desktop/LabeledDB_new/"
vertices, faces = off2numpy(path + 'Table/142.off')
mp.plot(vertices,faces);


# In[25]:


pca = PCA(n_components=2)
pca.fit(vertices)
pca.components_


# In[42]:


n_clusters=8
resolutions=np.array([10])
gain=np.array([0.35])
sigma=0.01

kmeans=KMeans(n_clusters=n_clusters)

params=tf.Variable(initial_value=np.array([[1],[1],[1]]).astype(np.float32)/np.sqrt(3),trainable=True)
X=tf.Variable(initial_value=vertices.astype(np.float32), trainable=False)

lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=1e-1, decay_steps=10, decay_rate=.1)
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr)


# In[43]:


f=tf.tensordot(X,params,axes=1)
mapper=MapperComplex(colors=f.numpy(),filters=f.numpy(),resolutions=resolutions, gains=gain, clustering=kmeans)
mapper.fit(vertices)
nt=mapper.get_pyvis()
nt.show('table2_initial.html')


# In[44]:


losses = []
K=10
with Parallel(n_jobs=-1) as parallel:
    for epoch in tqdm(range(200+1)):

        f=tf.tensordot(X,params,axes=1)
        fn=(f-tf.math.reduce_min(f))/(tf.math.reduce_max(f)-tf.math.reduce_min(f))
        scheme=smooth_scheme(fn.numpy(),resolutions,gain,sigma)

        upscheme=np.repeat(scheme,K,axis=0)

        assignments=bernoulli.rvs(upscheme)

        st,clusters=compute_mapper(X.numpy(),kmeans,assignments,maximum=8)
        
        with tf.GradientTape() as tape:
            f=tf.tensordot(X,params,axes=1)
            f_values=tf.repeat(tf.expand_dims(f,axis=0),clusters.shape[0],axis=0)

            f_values=tf.repeat(f_values,clusters.shape[2],axis=2)

            filtration=tf.math.reduce_sum(f_values*clusters,axis=1)/np.sum(clusters,axis=1)
            #l=[(_LowerStarSimplexTree(st[k],filtration[k],[0],11,False)[0][0]).reshape(-1,2) for k in range(n)]
            l=parallel(delayed(filter_st)(list(st[k].get_skeleton(1)),filtration.numpy()[k]) 
                for k in range(K))
            loss=0
            
            #dgm=tf.concat([tf.gather(filtration[k],l[k]) for k in range(K)],axis=0)
            for k in range(K):
                dgm=tf.gather(filtration[k],l[k]) 
                loss=loss-tf.math.reduce_sum(tf.math.abs((dgm[:,1]-dgm[:,0])))/K

            regularization = tf.math.square(tf.norm(params)-1)

            loss=loss+regularization

        gradients = tape.gradient(loss, [params])
        
        optimizer.apply_gradients(zip(gradients, [params]))


        losses.append(loss.numpy())


# In[45]:


plt.figure()
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[46]:


params


# In[47]:


tf.norm(params)


# In[48]:


tf.math.reduce_sum(params*np.array([[0],[1],[0]]))/tf.norm(params)


# In[49]:


mapper=MapperComplex(colors=f.numpy(),filters=f.numpy(),resolutions=resolutions, gains=gain, clustering=kmeans)
mapper.fit(vertices)
nt=mapper.get_pyvis()
nt.show('table2_final.html')


# In[ ]:




