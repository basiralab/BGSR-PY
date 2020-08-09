#  Details can be found in the original paper:Brain Graph Super-Resolution for Boosting Neurological
#  Disorder Diagnosis using Unsupervised Multi-Topology Residual Graph Manifold Learning.

#  This code requires Python3 to run.
#  For more details check BGSR.py file.
#  This is a translation into Python of BGSR MATLAB code. Link to MATLAB code: https://github.com/basiralab/BGSR
#  To test BGSR on random data, we defined the function 'simulateData_LR_HR' where the size of the dataset is chosen by the user.

#  ---------------------------------------------------------------------
#      Copyright 2020 Busra Asan (busraasan2@gmail.com), Istanbul Technical University.
#      Please cite the above paper if you use this code.
#      All rights reserved.
#      """
#
#   ------------------------------------------------------------------------------


import numpy as np
from simulate_Data_LR_HR import simulate_Data_LR_HR
from BGSR import BGSR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import math

mu1 = 0.8 # Mean parameter of the first Gaussian distribution
sigma1 = 0.4 # Standard deviation parameter of the first Gaussian distribution

mu2 = 0.7 # Mean parameter of the second Gaussian distribution
sigma2 = 0.6 # Standard deviation parameter of the second Gaussian distribution

kn = 10 # Number of selected features

np.seterr(divide='ignore', invalid='ignore')

# Functions for Pearson Correlation coded by @dfrankov in Stack Overflow.
def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def Leave_one_out_cross_validation(LR_average_Featurematrix, LR_data_av_x, LR_data_av_Labels, test_index, index):

    test_label = LR_data_av_Labels[test_index]
    test_data = LR_data_av_x[test_index][:][:]
    index = np.delete(index, test_index)
    train_labels = LR_data_av_Labels[index]
    train_data = LR_data_av_x[index][:][:]
    index = np.insert(index, test_index, test_index)

    return train_data, train_labels, test_data, test_label

LR_data_max_Featurematrix, LR_data_max_x, LR_data_max_Labels, LR_average_Featurematrix, LR_data_av_x, LR_data_av_Labels, HR_data_Featurematrix, HR_data_x, HR_data_Labels, c1, c2 = simulate_Data_LR_HR(mu1, sigma1, mu2, sigma2)

K1 = int(input("Select the number of neighbors for clusters: "))
while not K1:
    K1 = int(input('Please give a number: '))
while K1 > c1 or K1 > c2:
    K1 = int(input('Please choose a number smaller than number of groups: '))

K2 = int(input("Select the number of neighbors for fused similarity matrix construction: "))
while not K2:
    K2 = int(input('Please give a number: '))
while K2 > c1 or K1 > c2:
    K2 = int(input('Please choose a number smaller than number of groups: '))

#Initialization
index = np.arange(0, len(LR_data_av_Labels))
pHR_all = np.zeros((len(LR_data_av_Labels), len(LR_average_Featurematrix[1])))
GT_HR = np.zeros((len(HR_data_x[1]), len(HR_data_x[1])))

for test_index in range(0, len(LR_data_av_Labels)):

    train_data, train_labels, test_data, test_label = Leave_one_out_cross_validation(LR_average_Featurematrix, LR_data_av_x, LR_data_av_Labels, test_index, index)
    pHR = BGSR(train_data, train_labels, HR_data_Featurematrix, kn, K1, K2)
    for j in range(len(pHR)):
        pHR_all[test_index][j] = pHR[j]

# Display Ground truth of the last subject
GT_HR = HR_data_x[len(HR_data_Labels)-1][:][:]
fig1 = plt.figure(1)
plt.imshow(GT_HR)
plt.title(label="Ground truth HR")
#plt.show()

# Display the LR matrix of the last subject
GT_LR = LR_data_av_x[len(LR_data_av_Labels)-1][:][:]
fig2 = plt.figure(2)
plt.imshow(GT_LR)
plt.title(label="LR")

# Display of the predicted HR of the last subject
pred_HR = np.reshape(pHR, (len(GT_HR), len(GT_HR)))
pred_HR = pred_HR + np.transpose(pred_HR)
fig3 = plt.figure(3)
plt.imshow(pred_HR)
plt.title(label="Predicted HR")

# Display of the residual between predicted and GT HR the last subject
RES = np.abs(GT_HR - pred_HR)
fig4 = plt.figure(4)
plt.imshow(RES)
plt.title(label="Residual between predicted HR and GT HR")

MAE = mean_absolute_error(HR_data_Featurematrix, pHR_all)

z = np.zeros((1,1))
HR_vector = np.zeros((0, 1))
pHR_all_vector = np.zeros((0, 1))

# vectorizing HR_data_Featurematrix and pHR_all into 1D vector.
for ii in range(0, len(HR_data_Featurematrix[1])):
    for jj in range(0, len(HR_data_Featurematrix)):
        z[0,0] = HR_data_Featurematrix[jj,ii]
        HR_vector = np.append(HR_vector, z, axis=0)
        z[0,0] = pHR_all[jj,ii]
        pHR_all_vector = np.append(pHR_all_vector, z, axis=0)


PC = pearson_def(HR_vector, pHR_all_vector)

print("Mean Absolute Error:")
print(MAE)

print("Pearson Correlation:")
print(PC[0])

plt.show()
