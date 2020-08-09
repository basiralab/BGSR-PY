import snf
import numpy as np
import SIMLR_PY.SIMLR as SIMLR
# Estimation of the connectional brain template

def atlas(train_data, train_labels):
# Disentangling the heterogeneous distribution of the input_ networks using SIMLR clustering method
    z = np.zeros((1,1))
    k = np.zeros((len(train_labels), len(train_data[1])*len(train_data[1])))

    for i in range(0, len(train_labels)):

        k1 = train_data[i][:][:]
        k2 = np.zeros((0, 1))
        #vectorizing k1 into 1D vector k2.
        for ii in range(0, len(train_data[0])):
            for jj in range(0, len(train_data[0])):
                z[0,0] = k1[jj,ii]
                k2 = np.append(k2, z, axis=0)

        k2 = np.transpose(k2)

        for h in range(0, len(train_data[1])*len(train_data[1])):
            k[i][h] = k2[0][h]

    K = 4 # number of neighbors
    simlr = SIMLR.SIMLR_LARGE(2,K,0) # This is how we initialize an object for SIMLR. The first input is number of rank (clusters) and the second input is number of neighbors.The third one is an binary indicator whether to use memory-saving mode.You can turn it on when the number of cells are extremely large to save some memory but with the cost of efficiency.
    S1, F1, val1, ind1 = simlr.fit(k)
    y_pred_X1 = simlr.fast_minibatch_kmeans(F1,2)

    # After using SIMLR, we extract each cluster independently
    C1 = np.zeros((0, len(train_data[1]), len(train_data[1]))) # initialize cluster1
    C2 = np.zeros((0, len(train_data[1]), len(train_data[1]))) # initialize cluster2
    for y in range(0, len(train_labels)):
        if y_pred_X1[y] == 0:
            C1 = np.append(C1, np.abs([train_data[y][:][:]]), axis=0)
        elif y_pred_X1[y] == 1:
            C2 = np.append(C2, np.abs([train_data[y][:][:]]), axis=0)

    # For each cluster, we non-linearly diffuse and fuse all networks into a local cluster-specific CBT using SNF

    # Setting all the parameters.
    alpha = 0.5 #hyperparameter, usually (0.3~0.8)
    T = 20 #Number of Iterations, usually (10~20)

    def network_atlas(C, K, alpha, T):
        datap = []
        for i in range(len(C)):
            datap.append(C[i][:][:])
        affinity_networks = snf.make_affinity(datap, K=K, mu=alpha) #The first step in SNF is converting these data arrays into similarity (or "affinity") networks.
        fused_network_atlas_C = snf.snf(affinity_networks, K=K, alpha=alpha, t=T) # Once we have our similarity networks we can fuse them together.
        return fused_network_atlas_C

    if C1.shape[0] > 1:
        atlas_c1 = network_atlas(C1, K, alpha, T) # First cluster-specific CBT
    else:
        atlas_c1 = C1[0][:][:]

    if C2.shape[0] > 1:
        atlas_c2 = network_atlas(C2, K, alpha, T) # Second cluster-specific CBT
    else:
        atlas_c2 = C2[0][:][:]

    # SNF

    CBT = snf.snf([atlas_c1, atlas_c2], K=K, t=T) # Global connectional brain template
    return CBT
