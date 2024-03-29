import numpy as np
import itertools
from joblib import Parallel, delayed
import gudhi as gd
from scipy.stats import bernoulli
from scipy.linalg import block_diag



def smooth_scheme(filters,resolutions,gain,sigma):
    """
    Implements the smooth cover assignment scheme for multidimensional filters. The filter values must be normalized beforehand to have values in [0,1].

    Parameters:
        filters (numpy array of shape (num_points) x (num_filters)): filter functions 
            (sometimes called lenses) used to compute the cover. Each column of the numpy array 
            defines a scalar function having values in [0,1] defined on the input points.
        resolutions (numpy array of shape num_filters containing integers): resolution of 
            each filter function, ie number of intervals required to cover each filter image.
        gain (numpy array of shape num_filters containing doubles in [0,1]): gain of 
            each filter function, ie overlap percentage of the intervals covering each filter image.
        sigma (float): sigma (also denoted delta) parameter used to construct the smooth assignment scheme.
    
    Returns:
        scheme (numpy array of shape (num_filters) x (resolution) x (num_points)): Bernoulli parameters of the smooth assignment scheme.
    """
    num_pts,num_filters=filters.shape
    
    # Compute the endpoints of the cover intervals for all filters 
    column_indices=np.indices((num_filters,resolutions.max()))[1]

    steps=np.repeat((1/resolutions).reshape((num_filters,1)),resolutions.max(),axis=1)

    gains=np.repeat(gain.reshape((num_filters,1)),resolutions.max(),axis=1)

    epsilons=gains/(2-2*gains)*steps

    left_endpoints=steps*column_indices-epsilons
    A=np.repeat(left_endpoints.reshape((num_filters,resolutions.max(),1)),num_pts,axis=2)

    right_endpoints=steps*(column_indices+1)+epsilons
    B=np.repeat(right_endpoints.reshape((num_filters,resolutions.max(),1)),num_pts,axis=2)

    # Construct the filter_values array that 
    filter_values=np.repeat(filters.T.reshape(num_filters,1,num_pts),resolutions.max(),axis=1)
    
    # Compute the probability of each point being in each patch
    G1=1-1/(1-(filter_values-A)**2/sigma**2)
    G1[G1>0]=-np.inf
    G1=np.exp(G1)

    G2=1-1/(1-(filter_values-B)**2/sigma**2)
    G2[G2>0]=-np.inf
    G2=np.exp(G2)

    scheme=(filter_values<B)*(filter_values>A)+G1+G2
    scheme[scheme>1]=1
    
    return scheme


def cluster_patch(Binned_data,k,p_ind,X,clustering,maximum,input_type):
    """
    Clusters a patch. Used to compute multiple Mappers in parallel.

    Parameters:
        Binned_data (list of length num_mappers containing lists of length resolution): list containing the indexes of data points located in each patch for each Mapper
        k (integer): index of Mapper
        p_ind (integer): index of patch
        X (numpy array of shape (num_points) x (num_coordinates) if point cloud and (num_points) x (num_points) if distance matrix): input point cloud or distance matrix.
        clustering (class): clustering class. Common clustering classes can be found in the scikit-learn library (such as AgglomerativeClustering for instance).
        maximum (integer): maximum number of clusters possible in one patch.
        input_type (string): type of input data. Either "point cloud" or "distance matrix".
    
    Returns:
        clusters (numpy array of shape (n_samples,)): Index of the cluster each point in the patch belongs to.
        k (integer): index of Mapper
        p_ind (integer): index of patch
    """
    data_bin=Binned_data[k][p_ind]
    if len(data_bin) > 1:
        clusters = clustering.fit_predict(X[data_bin,:]) if input_type == "point cloud" \
        else clustering.fit_predict(X[data_bin,:][:,data_bin])
    elif len(data_bin) == 1:
        clusters = np.array([0])
    else:
        clusters = np.array([])
    clusters=clusters+p_ind*maximum
    return clusters,k,p_ind

def compute_mapper(X,clustering,assignments,input_type="point cloud",maximum=3):
    """
    Computes Mappers belonging to several cover assignments in Parallel. Only supports one dimensional filters.

    Parameters:
        X (numpy array of shape (num_points) x (num_coordinates) if point cloud and (num_points) x (num_points) if distance matrix): input point cloud or distance matrix.
        clustering (class): clustering class. Common clustering classes can be found in the scikit-learn library (such as AgglomerativeClustering for instance).
        assignments (numpy array of shape (num_mappers) x (resolution) x (num_points) containing values in {0,1}): Cover assignments for each Mapper.
        input_type (string): type of input data. Either "point cloud" or "distance matrix".
        maximum (integer): maximum number of clusters possible in one patch.
    
    Returns:
        simplex_trees (list of gd.SimplexTree() instances of length num_mappers): list of Mapper simplicial complexes represented by gudhi Simplex Trees. For computational efficiency considerations, empty clusters are present as isolated vertices in the simplicial complexes.
        clusters_array (numpy array of shape (num_mappers) x (num_points) x (maximum x resolution) containing values in {0,1}): Belongings, encoded as ones and zeros, of the points to the clusters in each Mapper.
    """
    K,resolution,num_pts=assignments.shape
    
    st0=gd.SimplexTree()
    st0.insert_batch((np.array(range(maximum*resolution))[np.newaxis,:]),np.zeros((maximum*resolution,)))
    
    # Initialize the list of simplex trees coreresponding to the Mappers and the clusters_array that gives the belongings of the points to the clusters in each Mapper.
    simplex_trees, clusters_array = [gd.SimplexTree(st0) for k in range(K)], np.zeros((K,num_pts,maximum*resolution))

    # Compute the Binned data map that associates each patch to the points that belong to it
    Binned_data=[[np.where(assignments[k,p,:]==1)[0] for p in range(resolution)] 
                 for k in range(K)]

    # Compute the clustering in each patch in parallel
    clusters_list = Parallel(n_jobs=-1)(delayed(cluster_patch)(Binned_data, k, p, X, clustering,maximum,input_type) for k,p in itertools.product(range(K),range(resolution)))

    # Fill the clusters_array
    for v in clusters_list :
        if v[0].size!=0:
            clusters_array[v[1],Binned_data[v[1]][v[2]],v[0]]=1 
    
    # Find clusters with non-empty intersection using matrix product
    M=np.matmul(np.transpose(clusters_array,axes=(0,2,1)),clusters_array)
    
    M[:,range(M.shape[1]),range(M.shape[1])]=0
    for k in range(K):
        M[k,np.triu_indices(M.shape[1])[0],np.triu_indices(M.shape[1])[1]]=0 
    
    # Fill the simplex trees
    for splx in zip(*np.where(M!=0)):
        simplex_trees[splx[0]].insert(splx[1:])
        
    return(simplex_trees,clusters_array)


def _LowerStarSimplexTree(simplextree, filtration, dimensions, homology_coeff_field, persistence_dim_max):
    # Parameters: simplextree (simplex tree on which to compute persistence)
    #             filtration (function values on the vertices of st),
    #             dimensions (homology dimensions),
    #             homology_coeff_field (homology field coefficient)
    
    simplextree.reset_filtration(-np.inf, 0)

    # Assign new filtration values
    for i in range(simplextree.num_vertices()):
        simplextree.assign_filtration([i], filtration[i])
    simplextree.make_filtration_non_decreasing()
    
    # Compute persistence diagram
    simplextree.compute_persistence(homology_coeff_field=homology_coeff_field, persistence_dim_max=persistence_dim_max)
    
    # Get vertex pairs for optimization. First, get all simplex pairs
    pairs = simplextree.lower_star_persistence_generators()
    
    L_indices = []
    for dimension in dimensions:
    
        finite_pairs = pairs[0][dimension] if len(pairs[0]) >= dimension+1 else np.empty(shape=[0,2])
        essential_pairs = pairs[1][dimension] if len(pairs[1]) >= dimension+1 else np.empty(shape=[0,1])
        
        finite_indices = np.array(finite_pairs.flatten(), dtype=np.int32)
        essential_indices = np.array(essential_pairs.flatten(), dtype=np.int32)

        L_indices.append((finite_indices, essential_indices))

    return L_indices

def filter_st(sklt,f):
    """
    Gives the filtration indices corresponding to the finite regular persistence diagram in dimension 0 for a Mapper complex.

    Parameters:
        sklt (list of tuples (simplex,)): one-skeleton of the simplex tree corresponding to the Mapper complex.
        f (numpy array of shape num_vertices): filtration value on the vertices of the Mapper.
    
    Returns:
        diagram (numpy array of shape (diagram_size,2)): filtration indices corresponding to the finite regular persistence diagram in dimension 0 of the Mapper complex.
    """
    st=gd.SimplexTree()
    for splx in sklt:
        st.insert(splx[0])
    a=_LowerStarSimplexTree(st,f,[0],11,False)[0][0]
    return(a.reshape(-1,2))


def get_extended_persistence_generators(simplextree):
    
    st=gd.SimplexTree(simplextree)
    filtration=[v[1] for v in st.get_skeleton(0)]
    dummy=len(filtration)
    dim=st.dimension()
    result=([[] for i in range(dim+1)],[])

    ## extend filtration
    st.extend_filtration()
    st.compute_persistence()
    
    ## get persistence pairs
    for pair in st.persistence_pairs():
        if len(pair[1]):

            pair_dim=len(pair[0])-1

            d1=dummy in pair[0]
            d2=dummy in pair[1]

            if d1: pair[0].remove(dummy)
            if d2: pair[1].remove(dummy)

            birth_list=[(-2*d1+1)*filtration[v] for v in pair[0]]
            birth=pair[0][birth_list.index(max(birth_list))]

            death_list=[(-2*d2+1)*filtration[v] for v in pair[1]]
            death=pair[1][death_list.index(max(death_list))]

            result[0][pair_dim].append([birth,death])

    for i in range(dim+1):
        result[0][i]=np.array(result[0][i])
    
    return(result)



# The parameters of the model are the vertex function values of the simplex tree.

def _ExtendedSimplexTree(simplextree, filtration, dimensions, homology_coeff_field):
    # Parameters: simplextree (simplex tree on which to compute persistence)
    #             filtration (function values on the vertices of st),
    #             dimensions (homology dimensions),
    #             homology_coeff_field (homology field coefficient)
    
    simplextree.reset_filtration(-np.inf, 0)

    # Assign new filtration values
    for i in range(simplextree.num_vertices()):
        simplextree.assign_filtration([i], filtration[i])
    simplextree.make_filtration_non_decreasing()

    # Get vertex pairs for optimization. First, get all simplex pairs
    pairs = get_extended_persistence_generators(simplextree)
    
    L_indices = []
    for dimension in dimensions:
    
        finite_pairs = pairs[0][dimension] if len(pairs[0]) >= dimension+1 else np.empty(shape=[0,2])
        essential_pairs = pairs[1][dimension] if len(pairs[1]) >= dimension+1 else np.empty(shape=[0,1])
        
        finite_indices = np.array(finite_pairs.flatten(), dtype=np.int32)
        essential_indices = np.array(essential_pairs.flatten(), dtype=np.int32)

        L_indices.append((finite_indices, essential_indices))

    return L_indices


def filter_extended_st(sklt,f):
    """
    Gives the filtration indices corresponding to the finite extended persistence diagram in dimension 0 and dimension 1 (stacked) for a Mapper complex.

    Parameters:
        sklt (list of tuples (simplex,)): one-skeleton of the simplex tree corresponding to the Mapper complex.
        f (numpy array of shape num_vertices): filtration value on the vertices of the Mapper.
    
    Returns:
        diagram (numpy array of shape (diagram_size,2)): filtration indices corresponding to the finite regular persistence diagram in dimension 0 and dimension 1 (stacked) of the Mapper complex.
    """
    st=gd.SimplexTree()
    for splx in sklt:
        st.insert(splx[0])
    a=_ExtendedSimplexTree(st,f,[0,1],11)
    return(np.vstack((a[0][0].reshape(-1,2),a[1][0].reshape(-1,2))))
