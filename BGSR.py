#  Main function of BGSR framework for brain graph super-resolution.
#  This code requires Python 3 to run.
#  Details can be found in the original paper:Brain Graph Super-Resolution for Boosting Neurological
#  Disorder Diagnosis using Unsupervised Multi-Topology Residual Graph Manifold Learning.
#
#
#    ---------------------------------------------------------------------
#
#      This file contains the implementation of the key steps of our BGSR framework:
#      (1) Estimation of a connectional brain template (CBT)
#      (2) Proposed CBT-guided graph super-resolution :
#
#                        [pHR] = BGSR(train_data,train_Labels,HR_features,kn)
#
#                  Inputs:
#
#                           train_data: ((n-1) × m × m) tensor stacking the symmetric matrices of the training subjects (LR
#                           graphs)
#                                       n the total number of subjects
#                                       m the number of nodes
#
#                           train_Labels: ((n-1) × 1) vector of training labels (e.g., -1, 1)
#
#                           HR_features:   (n × (m × m)) matrix stacking the source HR brain graph.
#
#                           Kn: Number of most similar LR training subjects.
#
#
#                  Outputs:
#                          pHR: (1 × (m × m)) vector stacking the predicted features of the testing subject.
#
#
#
#      To evaluate our framework we used Leave-One-Out cross validation strategy.
#
#
#
# To test BGSR on random data, we defined the function 'simulateData_LR_HR' where the size of the dataset is chosen by the user.
#  ---------------------------------------------------------------------
#      Copyright 2020 Busra Asan (busraasan2@gmail.com), Istanbul Technical University.
#      Please cite the above paper if you use this code.
#      All rights reserved.
#      """
#
#   ------------------------------------------------------------------------------

import numpy as np
import snf
import SIMLR_PY.SIMLR as SIMLR
from atlas import atlas
import networkx as nx

def BGSR(train_data,train_labels,HR_features,kn, K1, K2):

    #These are a reproduction of the closeness, degrees and isDirected functions of aeolianine since I couldn't find a compatible functions in python.
    def isDirected(adj):

        adj_transpose = np.transpose(adj)
        for i in range(len(adj)):
            for j in range(len(adj[0])):
                if adj[j][i] != adj_transpose[j][i]:
                    return True
        return False

    def degrees(adj):

        indeg = np.sum(adj, axis=1)
        outdeg = np.sum(np.transpose(adj), axis=0)
        if isDirected(adj):
            deg = indeg + outdeg #total degree
        else: #undirected graph: indeg=outdeg
            deg = indeg + np.transpose(np.diag(adj)) #add self-loops twice, if any

        return deg

    def closeness(G, adj):

        c = np.zeros((len(adj),1))
        all_sum = np.zeros((len(adj[1]),1))
        spl = nx.all_pairs_dijkstra_path_length(G, weight="weight")
        spl_list = list(spl)
        for i in range(len(adj[1])):
            spl_dict = spl_list[i][1]
            c[i] = 1 / sum(spl_dict.values())
        return c

    sz1, sz2, sz3 = train_data.shape

    # (1) Estimation of a connectional brain template (CBT)
    CBT = atlas(train_data, train_labels)

    # (2) Proposed CBT-guided graph super-resolution
    # Initialization
    c_degree = np.zeros((sz1, sz2))
    c_closeness = np.zeros((sz1, sz2))
    c_betweenness = np.zeros((sz1, sz2))
    residual = np.zeros((len(train_data), len(train_data[1]), len(train_data[1])))

    for i in range(sz1):

        residual[i][:][:] = np.abs(train_data[i][:][:] - CBT) #Residual brain graph
        G = nx.from_numpy_matrix(np.array(residual[i][:][:]))
        for j in range(0, sz2):
             c_degree[i][j] = degrees(residual[i][:][:])[j] #Degree matrix
             c_closeness[i][j] = closeness(G, residual[i][:][:])[j] #Closeness matrix
             c_betweenness[i][j] = nx.betweenness_centrality(G, weight=True)[j] #Betweenness matrix

    # Degree similarity matrix
    simlr1 = SIMLR.SIMLR_LARGE(1,K1,0) #The first input is number of rank (clusters) and the second input is number of neighbors.The third one is an binary indicator whether to use memory-saving mode.You can turn it on when the number of cells are extremely large to save some memory but with the cost of efficiency.
    S1, F1, val1, ind1 = simlr1.fit(c_degree)
    y_pred_X1 = simlr1.fast_minibatch_kmeans(F1,1)

    # Closeness similarity matrix
    simlr2 = SIMLR.SIMLR_LARGE(1,K1,0)
    S2, F2, val2, ind2 = simlr2.fit(c_closeness)
    y_pred_X2 = simlr2.fast_minibatch_kmeans(F2,1)

    # Betweenness similarity matrix
    if not np.count_nonzero(c_betweenness):
        S3 = np.zeros((len(c_betweenness), len(c_betweenness)))
    else:
        simlr3 = SIMLR.SIMLR_LARGE(1,K1,0)
        S3, F3, val3, ind3 = simlr3.fit(c_betweenness)
        y_pred_X3 = simlr3.fast_minibatch_kmeans(F3,1)

    alpha = 0.5 # hyperparameter, usually (0.3~0.8)
    T = 20  # Number of Iterations, usually (10~20)

    wp1 = snf.make_affinity(S1.toarray(), K=K2, mu=alpha)
    wp2 = snf.make_affinity(S2.toarray(), K=K2, mu=alpha)
    wp3 = snf.make_affinity(S3, K=K2, mu=alpha)
    FSM = snf.snf([wp1, wp2, wp3], K=K2, alpha=alpha, t=T) #Fused similarity matrix
    FSM_sorted = np.sort(FSM, axis=0)

    ind = np.zeros((kn,1))
    HR_ind = np.zeros((kn, len(HR_features[1])))
    for i in range(1, kn+1):
         a,b,pos = np.intersect1d(FSM_sorted[len(FSM_sorted)-i][0],FSM, return_indices=True)
         ind[i-1] = pos
         for j in range(len(HR_features[1])):
             HR_ind[i-1][j] = HR_features[int(ind[i-1][0])][j]

    pHR = np.mean(HR_ind, axis=0) # Predicted features of the testing subject
    return pHR
