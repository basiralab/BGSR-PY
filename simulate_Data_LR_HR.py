import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import xlsxwriter

mu1 = 0.8 #Mean parameter of the first Gaussian distribution
sigma1 = 0.4 #Standard deviation parameter of the first Gaussian distribution

mu2 = 0.7 # Mean parameter of the second Gaussian distribution
sigma2 = 0.6 #Standard deviation parameter of the second Gaussian distribution

def simulate_Data_LR_HR(mu1, sigma1, mu2, sigma2):

    Featurematrix = [];
    max_Featurematrix=[];
    AV_Featurematrix=[];

    c1 = int(input("Select the number of class 1 graphs: "))
    while not c1:
        c1 = int(input('Please give a number: '))
    while c1 < 5:
        c1 = int(input('Please choose a number > 4: '))

    c2 = int(input('Select the number of class 2 graphs: '))
    while not c2:
        c2 = int(input('Please give a number: '))
    while c2 < 5:
        c2 = int(input('Please choose a number > 4: '))

    m = int(input('Select the number of nodes (i.e. ROI\'s for brain graphs): '))
    while not m:
        m = int(input('Please give a number: '))
    while (m < 20) | (m%4 != 0):
        m = int(input('Please choose a number > 20 and a multiple of 4: '))

    N = c1 + c2
    datac1 = np.random.normal(mu1, sigma1, [c1, m, m]) # Normal random distribution
    datac2 = np.random.normal(mu2, sigma2, [c2, m, m]) # Normal random distribution
    data1 = np.concatenate((datac1, datac2), axis=0)

    # Creating a 1D vector from the entries of datac1 and datac2 vectors in order to draw a histogram graph.
    datac1_vector = datac1.flatten()
    datac2_vector = datac2.flatten()
    # Drawing samples from two different distributions to simulate both classes
    plt.hist(datac1_vector, bins = 115, edgecolor=(0, 1, 0, 1), fc=(0, 1, 0, 0.8))
    plt.hist(datac2_vector, bins = 115, edgecolor=(0, 0, 1, 0.4), fc=(0, 0, 1, 0.3))
    plt.show()

    V = np.zeros((int(m/4), 4, 4))
    AV = np.zeros((N, int(m/4), int(m/4)))
    maxi = np.zeros((N, int(m/4), int(m/4)))
    Featurematrix = np.zeros((0,m*m))
    z = np.zeros((1,1))

    for i in range(0,N):
        x = np.zeros((0,1))
        LR = np.zeros((int(m/4), 4, 4))
        data1[i][:][:] = np.squeeze(data1[i][:][:]) - np.diag(np.diag(np.squeeze(data1[i][:][:]))) #Eliminating self-symmetry
        data1[i][:][:] = np.true_divide((np.squeeze(data1[i][:][:]) + np.transpose(np.squeeze(data1[i][:][:]))),2) #Insure data symmetry
        t = np.triu(np.squeeze(data1[i,:,:]),1) #Upper triangular matrix constructed
        #t is vectorized:
        for ii in range(0,m):
            for jj in range(0,m):
                z[0,0] = t[jj,ii]
                x = np.append(x, z, axis=0)
        x1 = np.transpose(x) #transpose of x is x1
        Featurematrix = np.append(Featurematrix, x1, axis=0)

        jp=0
        jc=3
        r=-1
        H = np.zeros((int(m/4), int(m/4), 4, 4))
        while jp < m:
            r = r + 1
            for v in range(jp, jc+1, 4):
                ip = 0
                ic = 3
                o = 0
                while (ic < m):
                    mm = 0
                    for k in range(ip, ic+1):
                        n = 0
                        for l in range(jp, jc+1):
                            LR[o][mm][n] = np.squeeze(data1[i][k][l])
                            n = n+1
                        mm = mm+1
                    ip = ic+1
                    ic = ip+3
                    o = o+1
            for p in range(0, int(m/4)):
                for j in range(0, 4):
                    for s in range(0, 4):
                        H[r][p][j][s] = LR[p][j][s]
            jp = jc+1
            jc = jp+3


        for a1 in range(0, int(m/4)):

            for g in range(0, int(m/4)):
                for j in range(0, 4):
                    for s in range(0, 4):
                        V[g][j][s] = H[a1][g][j][s]

            for b1 in range(0,int(m/4)):
                q1 = np.squeeze(V[b1][:][:])
                maxi[i][b1][a1] = np.amax(np.amax(q1[:][:], axis=0), axis=0)
                AV[i][b1][a1] = np.mean(np.mean(q1, axis=0), axis = 0)

    def pooling_LR_data(max_av):
        t = np.triu(np.squeeze(max_av[i,:,:]),1) # Upper triangular part of matrix
        #vectorizing t (triangular matrix) into 1 column vector
        x = np.zeros((0,1))
        for ii in range(0,int(m/4)):
            for jj in range(0,int(m/4)):
                z[0,0] = t[jj,ii]
                x = np.append(x, z, axis=0)
        x = np.reshape(x, (1,len(x)))
        max_av_Featurematrix = np.zeros((0, len(t)*len(t)))
        max_av_Featurematrix = np.append(max_av_Featurematrix, x, axis=0)
        LR_data_x = max_av
        return max_av_Featurematrix, LR_data_x

    max_Featurematrix, LR_data_max_x = pooling_LR_data(maxi)
    LR_data_max_Featurematrix = Featurematrix
    LR_data_max_Labels = np.concatenate((np.ones((c1,1)),-1*np.ones((c2,1))), axis=0)

    AV_Featurematrix, LR_data_av_x = pooling_LR_data(AV)
    LR_average_Featurematrix = Featurematrix
    LR_data_av_Labels = np.concatenate((np.ones((c1,1)),-1*np.ones((c2,1))), axis=0)

    HR_data_Featurematrix = Featurematrix
    HR_data_x = data1
    HR_data_Labels = np.concatenate((np.ones((c1,1)),-1*np.ones((c2,1))), axis=0)

    return LR_data_max_Featurematrix, LR_data_max_x, LR_data_max_Labels, LR_average_Featurematrix, LR_data_av_x, LR_data_av_Labels, HR_data_Featurematrix, HR_data_x, HR_data_Labels, c1, c2
