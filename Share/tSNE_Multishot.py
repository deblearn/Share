#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as Math
import numpy as np
import pylab as Plot
import pylab as pl
import pickle as pkl
import time
import seaborn as sns
import itertools
import random


def Hbeta(D=Math.array([]), beta=1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = Math.exp(-D.copy() * beta);
    sumP = sum(P);
    H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
    P = P / sumP;
    return H, P;


def x2p(X=Math.array([]), tol=1e-5, perplexity= 100.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    print "Computing pairwise distances..."
    (n, d) = X.shape;
    sum_X = Math.sum(Math.square(X), 1);
    D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X);
    P = Math.zeros((n, n));
    beta = Math.ones((n, 1));
    logU = Math.log(perplexity);

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print "Computing P-values for point ", i, " of ", n, "..."

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -Math.inf;
        betamax = Math.inf;
        Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i + 1:n]))];
        (H, thisP) = Hbeta(Di, beta[i]);

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while Math.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy();
                if betamax == Math.inf or betamax == -Math.inf:
                    beta[i] = beta[i] * 2;
                else:
                    beta[i] = (beta[i] + betamax) / 2;
            else:
                betamax = beta[i].copy();
                if betamin == Math.inf or betamin == -Math.inf:
                    beta[i] = beta[i] / 2;
                else:
                    beta[i] = (beta[i] + betamin) / 2;

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i]);
            Hdiff = H - logU;
            tries = tries + 1;

        # Set the final row of P
        P[i, Math.concatenate((Math.r_[0:i], Math.r_[i + 1:n]))] = thisP;

    # Return final P-matrix
    print "Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta));
    return P;


def pca(X=Math.array([]), no_dims=50):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

    print "Preprocessing the data using PCA..."
    (n, d) = X.shape;
    X = X - Math.tile(Math.mean(X, 0), (n, 1));
    (l, M) = Math.linalg.eig(Math.dot(X.T, X));
    Y = Math.dot(X, M[:, 0:no_dims]);
    return Y;


def tsne(X=Math.array([]), no_dims=2, initial_dims=50, perplexity= 100.0):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if isinstance(no_dims, float):
        print "Error: array X should have type float.";
        return -1;
    if round(no_dims) != no_dims:
        print "Error: number of dimensions should be an integer.";
        return -1;

    # Initialize variables
    X = pca(X, initial_dims).real;
    (n, d) = X.shape;
    max_iter = 1000;
    initial_momentum = 0.5;
    final_momentum = 0.8;
    eta = 500;
    min_gain = 0.01;
    Y = Math.random.randn(n, no_dims);
    dY = Math.zeros((n, no_dims));
    iY = Math.zeros((n, no_dims));
    gains = Math.ones((n, no_dims));

    # Compute P-values
    P = x2p(X, 1e-5, perplexity);
    # (bbb, bbbb) = P.shape;
    P = P + Math.transpose(P);
    P = P / Math.sum(P);
    P = P * 4;  # early exaggeration
    P = Math.maximum(P, 1e-12);

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = Math.sum(Math.square(Y), 1);
        num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
        num[range(n), range(n)] = 0;
        Q = num / Math.sum(num);
        Q = Math.maximum(Q, 1e-12);

        # Compute gradient
        PQ = P - Q;
        for i in range(n):
            dY[i, :] = Math.sum(Math.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0);

        # Perform the update
        if iter < 250:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
        gains[gains < min_gain] = min_gain;
        iY = momentum * iY - eta * (gains * dY);
        Y = Y + iY;
        Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));
        # print(Y[1][0])
        # print(Y[2][1])

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = Math.sum(P * Math.log(P / Q));
            print "Iteration ", (iter + 1), ": error is ", C

        # Stop lying about P-values
        if iter == 990:
            P = P / 4;

    # Return solution
    return Y;


def updateS(Y, iY_Site_1, iY_Site_2, iY_Site_3):
    iY = (iY_Site_1 + iY_Site_2 + iY_Site_3) / 3
    Y = Y + iY

    return Y


def updateL(Y, G):
    return Y + G


def demeanS(Y, average_Y):
    return Y - Math.tile(average_Y, (Y.shape[0], 1))


def avgmean(Y1, Y2, Y3):
    avg_Y = (Math.mean(Y1, 0) + Math.mean(Y2, 0) + Math.mean(Y3, 0)) / 3
    return avg_Y;


def demeanL(Y, average_Y):
    # Y= (Y22 + Y23 + Y24)/3
    pp = Math.tile(Math.mean(Y, 0), (Y.shape[0], 1))
    return Y - Math.tile(average_Y, (Y.shape[0], 1))


def layout_plot(d, save=False, frame=0, path='./'):
    for k in d.keys():
        if k == 'shared': continue
        d[k]['data'] = d[k]['data'][d['shared']['data'].shape[0] - 1:]
        d[k]['labels'] = d[k]['labels'][d['shared']['data'].shape[0] - 1:]

    numbers = []
    for k in d.keys():
        numbers.extend(list(np.unique(d[k]['labels']).astype('int')))

    numbers = set(numbers)

    colors = {}
    clist = [(255, 255, 204),
             (255, 237, 160),
             (254, 217, 118),
             (254, 178, 76),
             (253, 141, 60),
             (252, 78, 42),
             (227, 26, 28),
             (189, 0, 38),
             (128, 0, 38),
             (0, 0, 0)]

    clist = [(166, 206, 227),
             (31, 120, 180),
             (178, 223, 138),
             (51, 160, 44),
             (251, 154, 153),
             (227, 26, 28),
             (253, 191, 111),
             (255, 127, 0),
             (202, 178, 214),
             (106, 61, 154)]

    cc = 0
    for c, p in zip(numbers, sns.color_palette("bright", len(numbers))):
        colors[c] = map(lambda x: x / 255., clist[cc])
        cc += 1

    pl.hold(True)
    markers = {'shared': 'o', 'site 1': 'D', 'site 2': 's', 'site 3': 'v'}
    for k in d.keys():
        x = d[k]['data'][:, 0]
        y = d[k]['data'][:, 1]
        c = d[k]['labels']
        pl.scatter(x, y, s=40,
                   c=[colors[x] for x in d[k]['labels']],
                   marker=markers[k], alpha=0.5, label=k)
        for l in np.unique(d[k]['labels']).astype('int'):
            lx, ly = np.mean(d[k]['data'][np.where(d[k]['labels'] == l)], 0)
            pl.text(lx, ly, k + '\n' + str(l) + ' ' + markers[k], bbox=dict(facecolor=colors[l], alpha=0.5))

    pl.axis('off')
    pl.legend(loc=0)
    if not save:
        pl.show()
    else:
        pl.savefig(path + 'frame_' + '%03d' % frame, transparent=True, pad_inches=0, bbox_inches='tight')
    pl.close()


def master(Y11, Shared_length, Site_1, Site1_length_X, Site_2, Site2_length_X, Site_3, Site3_length_X):
    max_iter = 1000;
    C_Site_1 = 0;
    C_Site_2 = 0;
    C_Site_3 = 0;
    Y22, dY_Site_1, iY_Site_1, gains_Site_1, P_Site_1, n_Site_1 = tsne1(Shared_length, Site1_length_X, Site_1, 2, 50,
                                                                        100.0, Y11);
    Y23, dY_Site_2, iY_Site_2, gains_Site_2, P_Site_2, n_Site_2 = tsne1(Shared_length, Site2_length_X, Site_2, 2, 50,
                                                                        100.0, Y11);
    Y24, dY_Site_3, iY_Site_3, gains_Site_3, P_Site_3, n_Site_3 = tsne1(Shared_length, Site3_length_X, Site_3, 2, 50,
                                                                        100.0, Y11);

    for iter in range(max_iter):
        Y22, iY_Site_1, Q_Site_1, C_Site_1, P_Site_1 = master_child(Y22, dY_Site_1, iY_Site_1, gains_Site_1, n_Site_1,
                                                                    Shared_length, P_Site_1, iter, C_Site_1)
        Y23, iY_Site_2, Q_Site_2, C_Site_2, P_Site_2 = master_child(Y23, dY_Site_2, iY_Site_2, gains_Site_2, n_Site_2,
                                                                    Shared_length, P_Site_2, iter, C_Site_2)
        Y24, iY_Site_3, Q_Site_3, C_Site_3, P_Site_3 = master_child(Y24, dY_Site_3, iY_Site_3, gains_Site_3, n_Site_3,
                                                                    Shared_length, P_Site_3, iter, C_Site_3)

        Y11 = updateS(Y11, iY_Site_1[:Shared_length, :], iY_Site_2[:Shared_length, :], iY_Site_3[:Shared_length, :])
        Y22[:Shared_length, :] = Y11
        Y23[:Shared_length, :] = Y11  # Y22[:Shared_length, :]
        Y24[:Shared_length, :] = Y11  # Y22[:Shared_length, :]

        Y22[Shared_length:, :] = updateL(Y22[Shared_length:, :], iY_Site_1[Shared_length:, :])
        Y23[Shared_length:, :] = updateL(Y23[Shared_length:, :], iY_Site_2[Shared_length:, :])
        Y24[Shared_length:, :] = updateL(Y24[Shared_length:, :], iY_Site_3[Shared_length:, :])

        # p=0;
        # Y[:Shared_length, :] = demeanS(Y[:Shared_length, :])

        # subtract mean values from Y values

        # average_Y = avgmean(Y22[Shared_length:, :],Y23[Shared_length:, :],Y24[Shared_length:, :])
        average_Y = avgmean(Y22, Y23, Y24)

        Y22[:Shared_length, :] = demeanS(Y22[:Shared_length, :], average_Y)
        Y23[:Shared_length, :] = Y22[:Shared_length, :]
        Y24[:Shared_length, :] = Y22[:Shared_length, :]

        Y22[Shared_length:, :] = demeanL(Y22[Shared_length:, :], average_Y)
        Y23[Shared_length:, :] = demeanL(Y23[Shared_length:, :], average_Y)
        Y24[Shared_length:, :] = demeanL(Y24[Shared_length:, :], average_Y)

        # Y22 = Y22 - Math.tile(Math.mean(Y, 0), (Y.shape[0], 1))
        # P_Site_3[deb5:max_value, :] = 0;

        Combined_Errors = (C_Site_1 + C_Site_2 + C_Site_3) / 3;
        C_Site_1 = Combined_Errors;
        C_Site_2 = Combined_Errors;
        C_Site_3 = Combined_Errors;
        print(Combined_Errors)

        Y11[:Shared_length_X, :] = Y22[:Shared_length_X, :]
        if (iter > 0):
            d = {'shared': {'data': Y11, 'labels': labels},
                 'site 1': {'data': Y22, 'labels': labels_1},
                 'site 2': {'data': Y23, 'labels': labels_2},
                 'site 3': {'data': Y24, 'labels': labels_3}}
            layout_plot(d, save=True, frame=iter, path='/home/deb/figure100/18th July/Final12/')

            # with open('/tmp/Deb/iter.pkl', 'wb') as f:
            #	pkl.dump(d, f)

            # time.sleep(5)

    # return (Y22,Y23,Y24)
    return (Y22, Y23, Y24, d)


def master_child(Y, dY, iY, gains, n, Shared_length, P, iter, C):
    # Compute pairwise affinities

    max_iter = 1000;
    initial_momentum = 0.5;
    final_momentum = 0.8;
    eta = 500;
    min_gain = 0.01;
    no_dims = 2;

    sum_Y = Math.sum(Math.square(Y), 1);
    num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
    num[range(n), range(n)] = 0;
    Q = num / Math.sum(num);
    Q = Math.maximum(Q, 1e-12);

    # Compute gradient
    PQ = P - Q;
    for i in range(n):
        dY[i, :] = Math.sum(Math.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0);

    # Perform the update
    if iter < 250:
        momentum = initial_momentum
    else:
        momentum = final_momentum
    gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
    gains[gains < min_gain] = min_gain;
    iY = momentum * iY - eta * (gains * dY);

    # Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));
    # print(Y[1][0])
    # print(Y[2][1])

    # Compute current value of cost function
    if (iter + 1) % 10 == 0:
        C = Math.sum(P * Math.log(P / Q));
        print "Iteration ", (iter + 1), ": error is ", C

    # Stop lying about P-values
    if iter == 990:
        P = P / 4;

    return (Y, iY, Q, C, P);


def tsne1(Shared_length, Site_length, X=Math.array([]), no_dims=2, initial_dims=50, perplexity= 100.0,
          Y_1=Math.array([])):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

    # Check inputs
    if isinstance(no_dims, float):
        print "Error: array X should have type float.";
        return -1;
    if round(no_dims) != no_dims:
        print "Error: number of dimensions should be an integer.";
        return -1;

    # Initialize variables
    # Y = [[0 for i in range(1778)] for j in range(2)]
    difference = Site_length - Shared_length;
    print(Shared_length, Site_length, difference)
    Y = Math.random.randn(Site_length, no_dims);
    Y[:Shared_length, :] = Y_1
    X = pca(X, initial_dims).real;
    (n, d) = X.shape;
    max_iter = 1000;
    initial_momentum = 0.5;
    final_momentum = 0.8;
    eta = 500;
    min_gain = 0.01;
    index = 0;
    index1 = 0;
    # Y_2 = Math.random.randn(difference, no_dims);

    # Y[:Shared_length,:] = Y_1
    # Y[Shared_length:,:] = Y_2

    dY = Math.zeros((n, no_dims));
    iY = Math.zeros((n, no_dims));
    gains = Math.ones((n, no_dims));

    # Compute P-values
    P = x2p(X, 1e-5, perplexity);
    # (bbb, bbbb) = P.shape;
    P = P + Math.transpose(P);
    P = P / Math.sum(P);
    P = P * 4;  # early exaggeration
    P = Math.maximum(P, 1e-12);
    (n1, d1) = P.shape;
    # Return solution
    return (Y, dY, iY, gains, P, n);


def normalize_columns(arr=Math.array([])):
    rows, cols = arr.shape
    for rows in xrange(rows):
        p = abs(arr[rows, :]).max()
        if (p != 0):
            arr[rows, :] = arr[rows, :] / abs(arr[rows, :]).max()

    return arr
def centralized_calculation_final(labels,Y,length,a,b):
    for i in range(0, length):
        if (labels[i] == 0):
            if ((Y[i][1] > a[0] or Y[i][1] < b[0]) or (Y[i][1] < a[0] or Y[i][1] > b[0])):
                Y[i][1] = random.uniform(a[0], b[0])
        if (labels[i] == 1):
            if ((Y[i][1] > a[1] or Y[i][1] < b[1]) or (Y[i][1] < a[1] or Y[i][1] > b[1])):
                Y[i][1] = random.uniform(a[1], b[1])
        if (labels[i] == 2):
            if ((Y[i][1] > a[2] or Y[i][1] < b[2]) or (Y[i][1] < a[2] or Y[i][1] > b[2])):
                Y[i][1] = random.uniform(a[2], b[2])
        if (labels[i] == 3):
            if ((Y[i][1] > a[3] or Y[i][1] < b[3]) or (Y[i][1] < a[3] or Y[i][1] > b[3])):
                Y[i][1] = random.uniform(a[3], b[3])
        if (labels[i] == 4):
            if ((Y[i][1] > a[4] or Y[i][1] < b[4]) or (Y[i][1] < a[4] or Y[i][1] > b[4])):
                Y[i][1] = random.uniform(a[4], b[4])
        if (labels[i] == 5):
            if ((Y[i][1] > a[5] or Y[i][1] < b[5]) or (Y[i][1] < a[5] or Y[i][1] > b[5])):
                Y[i][1] = random.uniform(a[5], b[5])
        if (labels[i] == 6):
            if ((Y[i][1] > a[6] or Y[i][1] < b[6]) or (Y[i][1] < a[6] or Y[i][1] > b[6])):
                Y[i][1] = random.uniform(a[6], b[6])
        if (labels[i] == 7):
            if ((Y[i][1] > a[7] or Y[i][1] < b[7]) or (Y[i][1] < a[7] or Y[i][1] > b[7])):
                Y[i][1] = random.uniform(a[7], b[7])
        if (labels[i] == 8):
            if ((Y[i][1] > a[8] or Y[i][1] < b[8]) or (Y[i][1] < a[8] or Y[i][1] > b[8])):
                Y[i][1] = random.uniform(a[8], b[8])
        if (labels[i] == 9):
            if ((Y[i][1] > a[9] or Y[i][1] < b[9]) or (Y[i][1] < a[9] or Y[i][1] > b[9])):
                Y[i][1] = random.uniform(a[9], b[9])
    return Y

def centralized_calculation(labels,Y,length,Total_sum,count):
    for i in range(0,length):
        if (labels[i] == 0):
            Total_sum[0] = Total_sum[0] + Y[i][1]
            count[0] += 1
        if (labels[i] == 1):
            Total_sum[1] = Total_sum[1] + Y[i][1]
            count[1] += 1
        if (labels[i] == 2):
            Total_sum[2] = Total_sum[2] + Y[i][1]
            count[2] += 1
        if (labels[i] == 3):
            Total_sum[3] = Total_sum[3] + Y[i][1]
            count[3] += 1
        if (labels[i] == 4):
            Total_sum[4] = Total_sum[4] + Y[i][1]
            count[4] += 1
        if (labels[i] == 5):
            Total_sum[5] = Total_sum[5] + Y[i][1]
            count[5] += 1
        if (labels[i] == 6):
            Total_sum[6] = Total_sum[6] + Y[i][1]
            count[6] += 1
        if (labels[i] == 7):
            Total_sum[7] = Total_sum[7] + Y[i][1]
            count[7] += 1
        if (labels[i] == 8):
            Total_sum[8] = Total_sum[8] + Y[i][1]
            count[8] += 1
        if (labels[i] == 9):
            Total_sum[9] = Total_sum[9] + Y[i][1]
            count[9] += 1
    return Total_sum,count


def centralized(Y11,Y22,Y23,Y24,labels,labels_1,labels_2,labels_3,Shared_length_X,Site1_length_X,Site2_length_X,Site3_length_X):
    Total_sum = [0]*100
    count = [0]*100
    a = [0]*100
    b = [0]*100

    Total_sum,count = centralized_calculation(labels,Y11,Shared_length_X,Total_sum,count)
    Total_sum, count = centralized_calculation(labels_1, Y22, Site1_length_X, Total_sum, count)
    Total_sum, count = centralized_calculation(labels_2, Y23, Site2_length_X, Total_sum, count)
    Total_sum, count = centralized_calculation(labels_3, Y24, Site3_length_X, Total_sum, count)


    for i in range(0,10):
        Total_sum[i] = Total_sum[i] / count[i]
        a[i] = Total_sum[i] + .15
        b[i] = Total_sum[i] - .15

    Y11= centralized_calculation_final(labels,Y11,Shared_length_X,a,b)
    Y22 = centralized_calculation_final(labels_1, Y22, Site1_length_X, a, b)
    Y23 = centralized_calculation_final(labels_2, Y23, Site2_length_X, a, b)
    Y24 = centralized_calculation_final(labels_3, Y24, Site3_length_X, a, b)

    return Y11,Y22,Y23,Y24




if __name__ == "__main__":
    print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
    print "Running example on 2,500 MNIST digits..."

    # X = Math.loadtxt("mnist2500_X.txt");
    # labels = Math.loadtxt("mnist2500_labels.txt");

    Combined_X = Math.loadtxt("Preprocessing_Mnist_X.txt");
    Combined_labels = Math.loadtxt("Preprocessing_label.txt");

    X = Math.loadtxt("Shared_Mnist_X.txt");
    labels = Math.loadtxt("Shared_lable.txt");
    Site_1 = Math.loadtxt("Site_1_Mnist_X.txt");
    labels_1 = Math.loadtxt("Site_1_Lable.txt");
    Site_2 = Math.loadtxt("Site_2_Mnist_X.txt");
    labels_2 = Math.loadtxt("Site_2_Lable.txt");
    Site_3 = Math.loadtxt("Site_3_Mnist_X.txt");
    labels_3 = Math.loadtxt("Site_3_Lable.txt");

    #X = normalize_columns(X1)
    #Site_1 = normalize_columns(Site_1_1)
    #Site_2 = normalize_columns(Site_2_2)
    #Site_3 = normalize_columns(Site_3_3)

    (Shared_length_X, Shared_length_Y) = X.shape;
    (Site1_length_X, Site1_length_Y) = Site_1.shape;
    (Site2_length_X, Site2_length_Y) = Site_2.shape;
    (Site3_length_X, Site3_length_Y) = Site_3.shape;

    # print(Shared_length_X, Shared_length_Y)
    # print(Site1_length_X, Site1_length_Y)
    # Y_Combined = tsne(Combined_X, 2, 20, 50.0)

    #Y11 = Math.random.randn(Shared_length_X, 2) #tsne(X, 2, 50, 100.0);
    Y11 = tsne(X, 2, 50, 100.0);

    # Modification starts
    Y22, Y23, Y24, d = master(Y11, Shared_length_X, Site_1, Site1_length_X, Site_2, Site2_length_X, Site_3,
                              Site3_length_X)
    Y11[:Shared_length_X, :] = Y22[:Shared_length_X, :]

    Y11,Y22,Y23,Y24 = centralized(Y11,Y22,Y23,Y24,labels,labels_1,labels_2,labels_3,Shared_length_X,Site1_length_X,Site2_length_X,Site3_length_X)

    Plot.scatter(Y11[:, 0], Y11[:, 1], 20, labels);
    Plot.scatter(Y22[:, 0], Y22[:, 1], 20, labels_1);
    Plot.scatter(Y23[:, 0], Y23[:, 1], 20, labels_2);
    Plot.scatter(Y24[:, 0], Y24[:, 1], 20, labels_3);
    # Plot.scatter(Y_Combined[:, 0], Y_Combined[:, 1], 20, Combined_labels);
    Plot.show();
