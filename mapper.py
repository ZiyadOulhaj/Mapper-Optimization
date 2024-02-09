import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from pyvis.network import Network
import networkx as nx
import matplotlib as mpl
import matplotlib.cm as cm
import itertools
import gudhi as gd
from joblib import Parallel, delayed

def to_hex(color):
    return '#'+''.join(["%02x" % e for e in color])

class MapperComplex(BaseEstimator, TransformerMixin):
    
    """
    This is a class for computing Mapper simplicial complexes on point clouds or distance matrices.
    """
    def __init__(self, input_type="point cloud", colors=None,
                       filters=None, filter_bnds=None, resolutions=None, gains=None, clustering=DBSCAN()):

        """
        Constructor for the MapperComplex class.

        Parameters:
            input_type (string): type of input data. Either "point cloud" or "distance matrix".
            colors (numpy array of shape (num_points) x (num_colors)): functions used to color the 
                nodes of the cover complex. More specifically, coloring is done by computing the 
                means of these functions on the subpopulations corresponding to each node. 
            filters (numpy array of shape (num_points) x (num_filters)): filter functions 
                (sometimes called lenses) used to compute the cover. Each column of the numpy array 
                defines a scalar function defined on the input points.
            filter_bnds (numpy array of shape (num_filters) x 2): limits of each filter, of the 
                form [[f_1^min, f_1^max], ..., [f_n^min, f_n^max]]. If one of the values is numpy.nan, 
                it can be computed from the dataset with the fit() method.
            resolutions (numpy array of shape num_filters containing integers): resolution of 
                each filter function, ie number of intervals required to cover each filter image.
            gains (numpy array of shape num_filters containing doubles in [0,1]): gain of 
                each filter function, ie overlap percentage of the intervals covering each filter image.
            clustering (class): clustering class (default sklearn.cluster.DBSCAN()). Common 
                clustering classes can be found in the scikit-learn library (such as 
                AgglomerativeClustering for instance).
        """
        self.filters, self.filter_bnds, self.resolutions = filters, filter_bnds, resolutions
        self.gains, self.colors, self.clustering = gains, colors, clustering
        self.input_type = input_type

    def get_pyvis(self, directed=False, cmap=cm.viridis):
        """
        Turn the 1-skeleton of the cover complex computed after calling fit() method into a pyvis graph.
        This function requires pyvis.

        Parameters:
            directed (bool): if True, a directed graph is returned. Arrows go from low filter values to high filter values
            cmap (matplotlib colormap): colormap used to represent the color attribute

        Returns:
            nt (pyvis graph): graph representing the 1-skeleton of the cover complex.
        """
        nt = Network('500px', '500px',notebook=True, directed=directed)
        st = self.simplex_tree
        c_vals=[float(node['colors']) for node in self.node_info.values()]
        norm = mpl.colors.Normalize(vmin=np.min(c_vals), vmax=np.max(c_vals))
        m = cm.ScalarMappable(norm=norm,  cmap=cmap)
        for (splx,_) in st.get_skeleton(0):
            nt.add_node(splx[0],color=to_hex(m.to_rgba(float(self.node_info[splx[0]]["colors"])
                                                   ,bytes=True)[0:3]))
        for (splx,_) in st.get_skeleton(1):
            if len(splx) == 2:
                if self.node_info[splx[0]]["colors"]<=self.node_info[splx[1]]["colors"]:
                    nt.add_edge(splx[0], splx[1])
                else:
                    nt.add_edge(splx[1], splx[0])
        return nt

    def fit(self, X, y=None):
        """
        Fit the MapperComplex class on a point cloud or a distance matrix: compute the Mapper complex 
        and store it in a simplex tree called simplex_tree.

        Parameters:
            X (numpy array of shape (num_points) x (num_coordinates) 
            if point cloud and (num_points) x (num_points) if distance matrix): input point 
            cloud or distance matrix.
            y (n x 1 array): point labels (unused).
        """
        num_pts, num_filters = self.filters.shape[0], self.filters.shape[1]

        # If some filter limits are unspecified, automatically compute them
        if self.filter_bnds is None:
            self.filter_bnds = np.hstack([np.min(self.filters, axis=0)[:,np.newaxis], 
                                          np.max(self.filters, axis=0)[:,np.newaxis]])
        # Initialize attributes
        self.simplex_tree, self.node_info = gd.SimplexTree(), {}
        
        # Compute the endpoints of the cover intervals for all filters 
        column_indices=np.indices((num_filters,self.resolutions.max()))[1]

        steps=np.repeat(((self.filter_bnds[:,1]-self.filter_bnds[:,0])/self.resolutions).reshape((num_filters,1)),self.resolutions.max(),axis=1)

        mins=np.repeat(self.filter_bnds[:,0].reshape((num_filters,1)),self.resolutions.max(),axis=1)

        gains=np.repeat(self.gains.reshape((num_filters,1)),self.resolutions.max(),axis=1)

        epsilons=gains/(2-2*gains)*steps

        left_endpoints=mins+steps*column_indices-epsilons

        right_endpoints=mins+steps*(column_indices+1)+epsilons
        
        # Find in which interval each point falls for each filter
        filter_values=np.repeat(self.filters.T.reshape(num_filters,1,num_pts),self.resolutions.max(),axis=1)

        comparison=(filter_values>=np.repeat(left_endpoints.reshape((num_filters,self.resolutions.max(),1)),num_pts,axis=2))*(filter_values<np.repeat(right_endpoints.reshape((num_filters,self.resolutions.max(),1)),num_pts,axis=2))

        # Compute the list of possible patches
        patch_list=[list(range(self.resolutions[f])) for f in range(num_filters)]
        patches=list(itertools.product(*patch_list))
        
        # Initialize the Binned data map that associates each patch to the points that belong to it
        # Initialize the cover map that associates each point to the clusters it belongs to
        Binned_data=[np.array([]) for p in patches]
        cover_map=[[] for pt in range(num_pts)]
        
        # Define the cluster_patch function that takes the Binned data map and the index of a patch and computes the list of clusters in that patch. It is used for parallel processing.
        def cluster_patch(Binned_data,p_ind):
            data_bin=Binned_data[p_ind]
            if len(data_bin) > 1:
                clusters = self.clustering.fit_predict(X[data_bin,:]) if self.input_type == "point cloud" \
                else self.clustering.fit_predict(X[data_bin,:][:,data_bin])
            elif len(data_bin) == 1:
                clusters = np.array([0])
            else:
                clusters = np.array([])

            return clusters
        
        # Fill the Binned data map
        for p_ind in range(len(patches)):

            p=patches[p_ind]
            point_list=[set(np.where(comparison[f,p[f],:])[0]) for f in range(num_filters)]

            Binned_data_current=point_list[0]
            for f in range(1,num_filters):
                Binned_data_current=Binned_data_current.intersection(point_list[f])
            Binned_data[p_ind]=np.array(list(Binned_data_current))

        
        # Compute the clustering in each patch in parallel
        clusters_list = Parallel(n_jobs=-1,backend="threading")(delayed(cluster_patch)(Binned_data, p_ind) for p_ind in range(len(patches)))
        
        # Go through the list of clusters
        current_max=0
        for clusters_ind in range(len(clusters_list)):
            # Change the name of the clusters to avoid confusion
            clusters_list[clusters_ind]=clusters_list[clusters_ind]+current_max
            clusters=clusters_list[clusters_ind]
            current_max+=len(np.unique(clusters))
            
            # Get information about each individual cluster
            for clus in np.unique(clusters):
                subpopulation = Binned_data[clusters_ind][clusters == clus]
                self.node_info[clus] = {}
                self.node_info[clus]["indices"] = subpopulation
                self.node_info[clus]["size"] = len(subpopulation)
                self.node_info[clus]["colors"] = np.mean(self.colors[subpopulation,:], axis=0)
                self.node_info[clus]["patch"] = patches[clusters_ind]
            # Fill the cover map
            [cover_map[pt].append(clus) for pt,clus in zip(Binned_data[clusters_ind], clusters)]

        # Insert the simplices into the Mapper
        for splx in cover_map:
            self.simplex_tree.insert(splx)



        return self
